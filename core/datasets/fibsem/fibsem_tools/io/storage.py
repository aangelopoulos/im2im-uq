import os

from zarr.meta import ZARR_FORMAT, json_dumps, json_loads
from zarr.storage import _prog_number
from zarr.storage import array_meta_key as zarr_array_meta_key
from zarr.storage import attrs_key as zarr_attrs_key
from zarr.storage import group_meta_key as zarr_group_meta_key
import fsspec
from zarr.storage import FSStore
from zarr.storage import normalize_storage_path
from zarr.n5 import (
    group_metadata_to_n5,
    group_metadata_to_zarr,
    array_metadata_to_n5,
    array_metadata_to_zarr,
    invert_chunk_coords,
    is_chunk_key,
    attrs_to_zarr,
    n5_keywords,
)
from zarr.meta import ZARR_FORMAT, json_dumps, json_loads
from zarr.errors import (
    FSPathExistNotDir,
    ReadOnlyError,
)
import re

from collections.abc import MutableMapping

array_meta_key = ".zarray"
group_meta_key = ".zgroup"
attrs_key = ".zattrs"
_prog_number = re.compile(r"^\d+$")


class FSStore(MutableMapping):
    """Wraps an fsspec.FSMap to give access to arbitrary filesystems

    Requires that ``fsspec`` is installed, as well as any additional
    requirements for the protocol chosen.

    This class is a modified version of `zarr.storage.FSStore` with changes that
    allow subclasses like `N5FSStore`.

    Parameters
    ----------
    url : str
        The destination to map. Should include protocol and path,
        like "s3://bucket/root"
    normalize_keys : bool
    key_separator : str
        Character to use when constructing the target path strings
        for data keys
    mode : str
        "w" for writable, "r" for read-only
    exceptions : list of Exception subclasses
        When accessing data, any of these exceptions will be treated
        as a missing key
    meta_keys : list or tuple of keys that are reserved for metadata.
        Defaults to the zarr defaults, i.e. (".zarray", ".zgroup", ".zattrs").
    storage_options : passed to the fsspec implementation
    """

    def __init__(
        self,
        url,
        normalize_keys=True,
        key_separator="/",
        mode="w",
        exceptions=(KeyError, PermissionError, IOError),
        meta_keys=(attrs_key, group_meta_key, array_meta_key),
        dimension_separator=None,
        **storage_options
    ):

        self.normalize_keys = normalize_keys
        self.key_separator = key_separator

        # For backwards compatibility. Guaranteed to be non-None
        if key_separator is not None:
            dimension_separator = key_separator

        self.key_separator = dimension_separator
        if self.key_separator is None:
            self.key_separator = "."

        # Pass attributes to array creation
        self._dimension_separator = dimension_separator

        protocol, _ = fsspec.core.split_protocol(url)
        # set auto_mkdir to True for local file system
        if protocol in (None, "file") and not storage_options.get("auto_mkdir"):
            storage_options["auto_mkdir"] = True
        self.map = fsspec.get_mapper(url, **storage_options)
        self.meta_keys = meta_keys
        self.fs = self.map.fs  # for direct operations
        self.path = self.fs._strip_protocol(url)
        self.mode = mode
        self.exceptions = exceptions

    def _normalize_key(self, key):
        key = normalize_storage_path(key).lstrip("/")
        if key:
            *bits, end = key.split("/")
            if end not in self.meta_keys:
                end = end.replace(".", self.key_separator)
                key = "/".join(bits + [end])

        return key.lower() if self.normalize_keys else key

    def getitems(self, keys, **kwargs):
        keys_transformed = [self._normalize_key(key) for key in keys]
        results = self.map.getitems(keys_transformed, on_error="omit")
        return {keys[keys_transformed.index(k)]: v for k, v in results.items()}

    def __getitem__(self, key):
        key = self._normalize_key(key)
        try:
            return self.map[key]
        except self.exceptions as e:
            raise KeyError(key) from e

    def setitems(self, values):
        if self.mode == "r":
            raise ReadOnlyError()
        values = {self._normalize_key(key): val for key, val in values.items()}
        self.map.setitems(values)

    def __setitem__(self, key, value):
        if self.mode == "r":
            raise ReadOnlyError()
        key = self._normalize_key(key)
        path = self.dir_path(key)
        try:
            if self.fs.isdir(path):
                self.fs.rm(path, recursive=True)
            self.map[key] = value
            self.fs.invalidate_cache(self.fs._parent(path))
        except self.exceptions as e:
            raise KeyError(key) from e

    def __delitem__(self, key):
        if self.mode == "r":
            raise ReadOnlyError()
        key = self._normalize_key(key)
        path = self.dir_path(key)
        if self.fs.isdir(path):
            self.fs.rm(path, recursive=True)
        else:
            del self.map[key]

    def delitems(self, keys):
        if self.mode == "r":
            raise ReadOnlyError
        # only remove the keys that exist in the store
        nkeys = [self._normalize_key(key) for key in keys if key in self]
        if len(nkeys) > 1:
            self.map.delitems(nkeys)

    def __contains__(self, key):
        key = self._normalize_key(key)
        return key in self.map

    def __eq__(self, other):
        return (
            type(self) == type(other)
            and self.map == other.map
            and self.mode == other.mode
        )

    def keys(self):
        return iter(self.map)

    def __iter__(self):
        return self.keys()

    def __len__(self):
        return len(list(self.keys()))

    def dir_path(self, path=None):
        store_path = normalize_storage_path(path)
        return self.map._key_to_str(store_path)

    def listdir(self, path=None):
        dir_path = self.dir_path(path)
        try:
            children = sorted(
                p.rstrip("/").rsplit("/", 1)[-1]
                for p in self.fs.ls(dir_path, detail=False)
            )
            if self.key_separator != "/":
                return children
            else:
                if array_meta_key in children:
                    # special handling of directories containing an array to map nested chunk
                    # keys back to standard chunk keys
                    new_children = []
                    root_path = self.dir_path(path)
                    for entry in children:
                        entry_path = os.path.join(root_path, entry)
                        if _prog_number.match(entry) and self.fs.isdir(entry_path):
                            for file_name in self.fs.find(entry_path):
                                file_path = os.path.join(dir_path, file_name)
                                rel_path = file_path.split(root_path)[1]
                                new_children.append(rel_path.replace(os.path.sep, "."))
                        else:
                            new_children.append(entry)
                    return sorted(new_children)
                else:
                    return children
        except IOError:
            return []

    def rmdir(self, path=None):
        if self.mode == "r":
            raise ReadOnlyError()
        store_path = self.dir_path(path)
        if self.fs.isdir(store_path):
            self.fs.rm(store_path, recursive=True)

    def getsize(self, path=None):
        store_path = self.dir_path(path)
        return self.fs.du(store_path, True, True)

    def clear(self):
        if self.mode == "r":
            raise ReadOnlyError()
        self.map.clear()


class N5FSStore(FSStore):
    """Implentation of the N5 format (https://github.com/saalfeldlab/n5) using `fsspec`,
    which allows storage on a variety of filesystems. Based on `zarr.N5Store`.

    Parameters
    ----------
    path : string
        Location of directory to use as the root of the storage hierarchy.
    normalize_keys : bool, optional
        If True, all store keys will be normalized to use lower case characters
        (e.g. 'foo' and 'FOO' will be treated as equivalent). This can be
        useful to avoid potential discrepancies between case-senstive and
        case-insensitive file system. Default value is False.

    Examples
    --------
    Store a single array::

        >>> import zarr
        >>> store = zarr.N5FSStore('data/array.n5')
        >>> z = zarr.zeros((10, 10), chunks=(5, 5), store=store, overwrite=True)
        >>> z[...] = 42

    Store a group::

        >>> store = zarr.N5FSStore('data/group.n5')
        >>> root = zarr.group(store=store, overwrite=True)
        >>> foo = root.create_group('foo')
        >>> bar = foo.zeros('bar', shape=(10, 10), chunks=(5, 5))
        >>> bar[...] = 42

    Notes
    -----

    This is an experimental feature.

    Safe to write in multiple threads or processes.

    """

    def __init__(self, *args, **kwargs):
        kwargs["key_separator"] = "/"
        kwargs["meta_keys"] = ("attributes.json",)
        super().__init__(*args, **kwargs)

    def _normalize_key(self, key):
        if is_chunk_key(key):
            key = invert_chunk_coords(key)

        key = normalize_storage_path(key).lstrip("/")
        if key:
            *bits, end = key.split("/")

            if end not in self.meta_keys:
                end = end.replace(".", self.key_separator)
                key = "/".join(bits + [end])
        return key.lower() if self.normalize_keys else key

    def __getitem__(self, key):
        if key.endswith(zarr_group_meta_key):

            key = key.replace(zarr_group_meta_key, self.meta_keys[0])
            value = group_metadata_to_zarr(self._load_n5_attrs(key))

            return json_dumps(value)

        elif key.endswith(zarr_array_meta_key):

            key = key.replace(zarr_array_meta_key, self.meta_keys[0])
            value = array_metadata_to_zarr(self._load_n5_attrs(key))

            return json_dumps(value)

        elif key.endswith(zarr_attrs_key):

            key = key.replace(zarr_attrs_key, self.meta_keys[0])
            value = attrs_to_zarr(self._load_n5_attrs(key))

            if len(value) == 0:
                raise KeyError(key)
            else:
                return json_dumps(value)
        return super().__getitem__(key)

    def __setitem__(self, key, value):
        if key.endswith(zarr_group_meta_key):

            key = key.replace(zarr_group_meta_key, self.meta_keys[0])

            n5_attrs = self._load_n5_attrs(key)
            n5_attrs.update(**group_metadata_to_n5(json_loads(value)))

            value = json_dumps(n5_attrs)

        elif key.endswith(zarr_array_meta_key):

            key = key.replace(zarr_array_meta_key, self.meta_keys[0])

            n5_attrs = self._load_n5_attrs(key)
            n5_attrs.update(**array_metadata_to_n5(json_loads(value)))

            value = json_dumps(n5_attrs)

        elif key.endswith(zarr_attrs_key):

            key = key.replace(zarr_attrs_key, self.meta_keys[0])

            n5_attrs = self._load_n5_attrs(key)
            zarr_attrs = json_loads(value)

            for k in n5_keywords:
                if k in zarr_attrs.keys():
                    raise ValueError(
                        "Can not set attribute %s, this is a reserved N5 keyword" % k
                    )

            # replace previous user attributes
            for k in list(n5_attrs.keys()):
                if k not in n5_keywords:
                    del n5_attrs[k]

            # add new user attributes
            n5_attrs.update(**zarr_attrs)

            value = json_dumps(n5_attrs)

        super().__setitem__(key, value)

    def __delitem__(self, key):

        if key.endswith(zarr_group_meta_key):  # pragma: no cover
            key = key.replace(zarr_group_meta_key, self.meta_keys[0])
        elif key.endswith(zarr_array_meta_key):  # pragma: no cover
            key = key.replace(zarr_array_meta_key, self.meta_keys[0])
        elif key.endswith(zarr_attrs_key):  # pragma: no cover
            key = key.replace(zarr_attrs_key, self.meta_keys[0])

        super().__delitem__(key)

    def __contains__(self, key):
        if key.endswith(zarr_group_meta_key):

            key = key.replace(zarr_group_meta_key, self.meta_keys[0])
            if key not in self:
                return False
            # group if not a dataset (attributes do not contain 'dimensions')
            return "dimensions" not in self._load_n5_attrs(key)

        elif key.endswith(zarr_array_meta_key):

            key = key.replace(zarr_array_meta_key, self.meta_keys[0])
            # array if attributes contain 'dimensions'
            return "dimensions" in self._load_n5_attrs(key)

        elif key.endswith(zarr_attrs_key):

            key = key.replace(zarr_attrs_key, self.meta_keys[0])
            return self._contains_attrs(key)

        return super().__contains__(key)

    def __eq__(self, other):
        return isinstance(other, N5FSStore) and self.path == other.path

    def listdir(self, path=None):

        if path is not None:
            path = invert_chunk_coords(path)

        # We can't use NestedDirectoryStore's listdir, as it requires
        # array_meta_key to be present in array directories, which this store
        # doesn't provide.
        children = super().listdir(path=path)
        if self._is_array(path):

            # replace n5 attribute file with respective zarr attribute files
            children.remove(self.meta_keys[0])
            children.append(zarr_array_meta_key)
            if self._contains_attrs(path):
                children.append(zarr_attrs_key)

            # special handling of directories containing an array to map
            # inverted nested chunk keys back to standard chunk keys
            new_children = []
            root_path = self.dir_path(path)
            for entry in children:
                entry_path = os.path.join(root_path, entry)
                if _prog_number.match(entry) and self.fs.isdir(entry_path):
                    for dir_path, _, file_names in self.fs.walk(entry_path):
                        for file_name in file_names:
                            file_path = os.path.join(dir_path, file_name)
                            rel_path = file_path.split(root_path + os.path.sep)[1]
                            new_child = rel_path.replace(os.path.sep, ".")
                            new_children.append(invert_chunk_coords(new_child))
                else:
                    new_children.append(entry)

            return sorted(new_children)

        elif self._is_group(path):

            # replace n5 attribute file with respective zarr attribute files
            children.remove(self.meta_keys[0])
            children.append(zarr_group_meta_key)
            if self._contains_attrs(path):  # pragma: no cover
                children.append(zarr_attrs_key)

            return sorted(children)

        else:

            return children

    def _load_n5_attrs(self, path):
        try:
            s = super().__getitem__(path)
            return json_loads(s)
        except KeyError:
            return {}

    def _is_group(self, path):

        if path is None:
            attrs_key = self.meta_keys[0]
        else:
            attrs_key = os.path.join(path, self.meta_keys[0])

        n5_attrs = self._load_n5_attrs(attrs_key)
        return len(n5_attrs) > 0 and "dimensions" not in n5_attrs

    def _is_array(self, path):

        if path is None:
            attrs_key = self.meta_keys[0]
        else:
            attrs_key = os.path.join(path, self.meta_keys[0])

        return "dimensions" in self._load_n5_attrs(attrs_key)

    def _contains_attrs(self, path):

        if path is None:
            attrs_key = self.meta_keys[0]
        else:
            if not path.endswith(self.meta_keys[0]):
                attrs_key = os.path.join(path, self.meta_keys[0])
            else:  # pragma: no cover
                attrs_key = path

        attrs = attrs_to_zarr(self._load_n5_attrs(attrs_key))
        return len(attrs) > 0
