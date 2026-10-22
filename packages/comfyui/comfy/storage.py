import functools
import logging
import os
import platform
import re

import comfy_aimdo.storage
from .cli_args import args


SYSFS = "/sys"
PROC_SELF_MOUNTINFO = "/proc/self/mountinfo"

_NVME_NAMESPACE = re.compile(r"^nvme\d+n\d+$")
_MOUNTINFO_ESCAPE = re.compile(r"\\([0-7]{3})")


def _read(path):
    try:
        with open(path, encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return None


def _sys(*parts):
    return os.path.join(SYSFS, *parts)


def _physical_block_devices(name):
    if os.path.exists(_sys("class", "block", name, "partition")):
        name = os.path.basename(os.path.dirname(os.path.realpath(_sys("class", "block", name))))

    slaves = _sys("class", "block", name, "slaves")
    try:
        children = os.listdir(slaves)
    except OSError:
        children = []
    if children:
        devices = []
        for child in children:
            devices.extend(_physical_block_devices(child))
        return devices
    return [name]


def _nvme_controllers(namespace):
    """Controllers carrying an NVMe namespace. With native NVMe multipath the
    namespace is named after its subsystem, not a controller, and each path
    device under ``multipath/`` lives in the directory of its controller."""
    block = _sys("class", "block", namespace)
    try:
        paths = os.listdir(os.path.join(block, "multipath"))
    except OSError:
        paths = []
    if paths:
        return sorted(os.path.basename(os.path.dirname(os.path.realpath(os.path.join(block, "multipath", p)))) for p in paths)
    device = os.path.join(block, "device")
    if not os.path.exists(device):
        return []
    return [os.path.basename(os.path.realpath(device))]


def _fast_nvme_controller(controller):
    speed = _read(_sys("class", "nvme", controller, "device", "current_link_speed"))
    width = _read(_sys("class", "nvme", controller, "device", "current_link_width"))
    if speed is None or width is None:
        return None
    try:
        speed_gts = float(speed.split()[0])
        width = int(width)
    except ValueError:
        return None
    return (speed_gts >= 8.0 and width >= 4) or (speed_gts >= 32.0 and width >= 2)


def _fast_nvme(name):
    if _NVME_NAMESPACE.match(name) is None:
        return False
    controllers = _nvme_controllers(name)
    if not controllers:
        return None
    results = [_fast_nvme_controller(controller) for controller in controllers]
    if any(result is None for result in results):
        return None
    return all(results)


def _unescape_mountinfo(field):
    return _MOUNTINFO_ESCAPE.sub(lambda match: chr(int(match.group(1), 8)), field)


def _within(mount_point, path):
    return mount_point == "/" or path == mount_point or path.startswith(mount_point + "/")


def _mount_of(path, device):
    """(fstype, source) of the mount holding ``path``, from mountinfo.

    The mount whose major:minor is ``st_dev`` is exact. btrfs gives every
    subvolume an anonymous device, and a subvolume reached through its parent
    rather than mounted itself (0:31 under / on appmana-001) has no entry of
    its own, so otherwise the mount is the one covering ``path``: each later
    mount covering ``path`` either sits on top of or shadows the earlier one.
    """
    raw = _read(PROC_SELF_MOUNTINFO)
    if raw is None:
        return None
    dev = f"{os.major(device)}:{os.minor(device)}"
    exact = None
    covering = None
    for line in raw.splitlines():
        head, separator, tail = line.partition(" - ")
        fields = head.split()
        tail_fields = tail.split()
        if not separator or len(fields) < 5 or len(tail_fields) < 2:
            continue
        mount = (tail_fields[0], _unescape_mountinfo(tail_fields[1]))
        if fields[2] == dev and exact is None:
            exact = mount
        if _within(_unescape_mountinfo(fields[4]), path):
            covering = mount
    return exact or covering


def _block_device_name(source):
    if not source.startswith("/"):
        return None
    try:
        rdev = os.stat(source).st_rdev
    except OSError:
        rdev = 0
    if rdev:
        sys_device = _sys("dev", "block", f"{os.major(rdev)}:{os.minor(rdev)}")
        if os.path.exists(sys_device):
            return os.path.basename(os.path.realpath(sys_device))
    name = os.path.basename(os.path.realpath(source))
    return name if os.path.exists(_sys("class", "block", name)) else None


def _btrfs_devices(name):
    """Every device of the btrfs filesystem ``name`` belongs to."""
    root = _sys("fs", "btrfs")
    try:
        filesystems = sorted(os.listdir(root))
    except OSError:
        return [name]
    for filesystem in filesystems:
        try:
            devices = os.listdir(os.path.join(root, filesystem, "devices"))
        except OSError:
            continue
        if name in devices:
            return sorted(devices)
    return [name]


def _backing_block_devices(path, device):
    sys_device = _sys("dev", "block", f"{os.major(device)}:{os.minor(device)}")
    if os.path.exists(sys_device):
        return [os.path.basename(os.path.realpath(sys_device))]
    mount = _mount_of(path, device)
    if mount is None:
        return None
    fstype, source = mount
    name = _block_device_name(source)
    if name is None:
        return None
    if fstype == "btrfs":
        return _btrfs_devices(name)
    return [name]


@functools.lru_cache(maxsize=None)
def _linux_fast_storage(device, path):
    names = _backing_block_devices(path, device)
    if names is None:
        return None
    devices = [physical for name in names for physical in _physical_block_devices(name)]
    results = [_fast_nvme(x) for x in devices]
    if any(x is None for x in results):
        return None
    return all(results)


def fast_storage(path):
    system = platform.system()
    if system == "Linux":
        path = os.path.realpath(path)
        try:
            device = os.stat(path).st_dev
        except OSError:
            return None
        return _linux_fast_storage(device, path)
    if system == "Windows":
        return comfy_aimdo.storage.fast_disk(path)
    return None


def annotate_state_dict(state_dict, path):
    path = os.path.realpath(path)
    for value in state_dict.values():
        untyped_storage = getattr(value, "untyped_storage", None)
        if untyped_storage is not None:
            untyped_storage()._comfy_source_path = path


def state_dict_fast_disk(state_dict):
    state_dicts = state_dict if isinstance(state_dict, (list, tuple)) else (state_dict,)
    paths = set()
    for sd in state_dicts:
        for value in sd.values():
            untyped_storage = getattr(value, "untyped_storage", None)
            if untyped_storage is not None:
                path = getattr(untyped_storage(), "_comfy_source_path", None)
                if path is not None:
                    paths.add(path)
    return model_fast_disk(sorted(paths))


def model_fast_disk(paths):
    if args.fast_disk or args.disable_fast_disk:
        fast = not args.disable_fast_disk
    else:
        results = [fast_storage(path) for path in paths]
        fast = bool(results) and all(result is True for result in results)
    logging.info("Model storage policy: fast_disk=%s paths=%s", fast, [os.path.realpath(path) for path in paths])
    return fast
