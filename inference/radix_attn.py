from __future__ import annotations
import torch
from dataclasses import dataclass
from typing import Dict
import time


def get_mha_cache_per_page(head_dim, num_kv_heads,  num_layers, page_size=1, tp_size=1, dtype_size=2):
    """
    :param head_dim: dim大小
    :param num_kv_heads: 头数
    :param num_layers: 模型层数
    :param page_size: 页大小，默认1
    :param tp_size: 并行策略的TP大小， 默认1为没有TP切分
    :param dtype_size: 数据大小，torch.float32 4字节，torch.float16/torch.bfloat16: 2字节
    :return:
    """
    size = 2 * num_kv_heads * head_dim * num_layers * page_size / tp_size * dtype_size
    return size


def get_num_pages_for_kv_cache(available_memory, cache_per_page):
    num_pages = int(available_memory // cache_per_page)
    kv_size = num_pages * cache_per_page / (1024**3)
    print(f"Allocating {num_pages} pages for KV cache, K + V = {kv_size:.3f}GB")
    return num_pages


def create_page_table(max_running_req, max_seq_len):
    page_table = torch.zeros((max_running_req, max_seq_len), dtype=torch.int32)
    return page_table


# 模型参数定义：
head_dim = 256
num_kv_heads = 8
num_layers = 1

# 可用显存大小
available_memory = 4  # 4G
print(f"可用显存大小{available_memory} GB")

cache_per_page = get_mha_cache_per_page(head_dim, num_kv_heads,  num_layers)
num_pages = get_num_pages_for_kv_cache(available_memory * (1024**3), cache_per_page)


# 定义一个简单请求格式：
@dataclass
class SimpleRequest:
    uid: int
    table_idx: int
    len: int


def demo():
    # 设定一个 num_pages大小用于演示
    num_pages = 50

    # 物理显存模拟：
    free_slots = torch.arange(num_pages,  dtype=torch.int32)

    # 构建page_table:
    page_table = create_page_table(5, 10)

    print("当前空余slots：")
    print(free_slots)

    # 定义显存的申请与释放函数：
    def allocate(req):
        nonlocal free_slots
        page_table[req.table_idx][:req.len] = free_slots[:req.len]
        free_slots = free_slots[req.len:]

    def free(req):
        nonlocal free_slots
        free_slots = torch.cat([free_slots, page_table[req.table_idx][:req.len]])

    req_0 = SimpleRequest(0, 0, 7)
    allocate(req_0)
    print("请求0 slots使用情况：")
    print(page_table[req_0.table_idx][:req_0.len])

    req_1 = SimpleRequest(1, 1, 7)
    allocate(req_1)
    print("请求1 slots使用情况：")
    print(page_table[req_1.table_idx][:req_1.len])
    print("="* 80)

    print("当前空余slots：")
    print(free_slots)
    print("="* 80)
    free(req_0)
    print("释放请求0后空余slots：")
    print(free_slots)

demo()



# 定义一个辅助函数：
def find_first_diff_pos(tensor1: torch.Tensor, tensor2: torch.Tensor) -> int:
    """
    比较两个一维PyTorch tensor的值，返回第一个不同值的位置
    Returns:
        int: 第一个不同值的索引位置。如果完全相同返回-1
    """
    if tensor1.dim() != 1 or tensor2.dim() != 1:
        raise ValueError("两个tensor都必须是一维的")

    len1, len2 = len(tensor1), len(tensor2)

    # 遍历两个tensor直到较短的那个结束
    for i in range(min(len1, len2)):
        if tensor1[i] != tensor2[i]:
            return i

    return min(len1, len2)


class RadixTreeNode:
    counter: int = 0

    def __init__(self, tic: int | None = None) -> None:
        self.children: Dict[int, RadixTreeNode] = {}
        self._parent: RadixTreeNode | None = None
        self.ref_count: int = 0
        self.uuid = RadixTreeNode.counter
        RadixTreeNode.counter += 1
        self.timestamp = tic or time.monotonic_ns()

        self._token_ids: torch.Tensor
        self._slots: torch.Tensor
        self._length: int

    def set_ids_slots(self, token_ids: torch.Tensor, slots: torch.Tensor) -> None:
        assert len(token_ids) == len(slots)
        self._token_ids = token_ids
        self._slots = slots
        self._length = len(token_ids)

    def set_parent(self, parent: RadixTreeNode) -> None:
        self._parent = parent
        parent.children[int(self._token_ids[0].item())] = self

    @property
    def length(self) -> int:
        return self._length

    @property
    def parent(self) -> RadixTreeNode:
        assert self._parent is not None
        return self._parent

    @property
    def slots(self) -> torch.Tensor:
        return self._slots

    def is_root(self) -> bool:
        return self._parent is None

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def get_match_len(self, input_ids: torch.Tensor) -> int:
        return find_first_diff_pos(self._token_ids, input_ids)

    def _split_at(self, pos: int) -> RadixTreeNode:
        assert 0 < pos < self.length
        parent = self.parent

        new_node = RadixTreeNode(self.timestamp)
        new_node.set_ids_slots(self._token_ids[:pos], self._slots[:pos])
        new_node.set_parent(parent)
        new_node.ref_count = self.ref_count

        self.set_ids_slots(self._token_ids[pos:], self._slots[pos:])
        self.set_parent(new_node)

        return new_node

    def __lt__(self, other: RadixTreeNode) -> bool:
        return self.timestamp < other.timestamp

    def __repr__(self) -> str:
        return f"RadixTreeNode(uuid={self.uuid}, tokens={self._token_ids.tolist()}, slots={self._slots.tolist()})"


def print_radix_tree(root: RadixTreeNode, max_depth: int = 10,
                    show_ref_count: bool = True, show_timestamp: bool = False) -> None:
    """
    以树形结构显示RadixTreeNode的形状

    Args:
        root: 根节点
        max_depth: 最大显示深度
        show_ref_count: 是否显示引用计数
        show_timestamp: 是否显示时间戳
    """

    def _print_node(node: RadixTreeNode, depth: int, prefix: str, is_last: bool = True) -> None:
        """递归打印节点及其子节点"""
        if depth > max_depth:
            return

        connector = "└── " if is_last else "├── "
        node_info = f"uuid={node.uuid}"
        if hasattr(node, '_token_ids') and node._token_ids is not None:
            token_str = str(node._token_ids.tolist())[:30]  # 限制长度
            node_info += f", tokens={token_str}"
        if hasattr(node, '_slots') and node._slots is not None:
            slot_str = str(node._slots.tolist())[:30]  # 限制长度
            node_info += f", slots={slot_str}"
        if show_ref_count:
            node_info += f", ref={node.ref_count}"

        if show_timestamp:
            node_info += f", ts={node.timestamp}"

        if node.is_leaf():
            node_info += " [L]"
        elif node.is_root():
            node_info += " [R]"

        print(f"{prefix}{connector}{node_info}")
        new_prefix = prefix + ("    " if is_last else "│   ")

        # 递归打印子节点
        child_count = len(node.children)
        for i, (key, child_node) in enumerate(sorted(node.children.items())):
            is_last_child = (i == child_count - 1)
            _print_node(child_node, depth + 1, new_prefix, is_last_child)

    print("\n" + "="*80)
    print("RADIX TREE STRUCTURE")
    print("="*80)

    if root.is_root():
        print("Root Node:")

    _print_node(root, 0, "")


from typing import Tuple

# 查找匹配前缀，有相同前缀，则分裂节点。
def walk(input_ids: torch.Tensor) -> Tuple[RadixTreeNode, int]:
    """
    返回值 node：匹配到的前缀node。
        prefix_len: 匹配长度
    """
    node = root_node
    prefix_len = 0
    indice_len = len(input_ids)
    while prefix_len < indice_len:
        this_id = int(req_1_ids[prefix_len].item())
        if this_id not in node.children:
            return node, prefix_len

        node = node.children[this_id]
        match_len = node.get_match_len(input_ids[prefix_len:])
        prefix_len += match_len

        if match_len != node.length:
            node = node._split_at(match_len)
            return node, prefix_len
    return node, prefix_len


RadixTreeNode.counter = 0 # 从0开始计数
root_node = RadixTreeNode()
root_node.ref_count = 1
node_0 = RadixTreeNode()
req_0_ids = torch.tensor([1, 3, 6, 7, 9, 77])
req_0_slots = torch.tensor([0, 1, 2, 3, 4, 7])

req_1_ids = torch.tensor([1, 3, 6, 7, 87, 66])
req_1_slots = torch.tensor([0, 1, 2, 3, 5, 6])

# 创建节点0
node_0.set_ids_slots(req_0_ids, req_0_slots)
node_0.set_parent(root_node)

# 打印插入node_0的状态：
print_radix_tree(root_node)

# 查找，并触发分裂操作
node, prefix_len = walk(req_1_ids)

# 增加一个处理句柄，记录request引用过的prefix
cache_handle = node

# 创建节点1
new_node = RadixTreeNode()
new_node.set_ids_slots(req_1_ids[prefix_len:], req_1_slots[prefix_len:].clone())
new_node.set_parent(cache_handle)

print_radix_tree(root_node)




import heapq

# 需要3个tokens空间
target_size = 3

# 查找叶子结点：
nodes = [root_node]
leave_nodes = []

while len(nodes) > 0:
    node = nodes.pop()
    if node.is_leaf():
        if node.ref_count == 0:
            leave_nodes.append(node)
    else:
        for child in node.children.values():
            nodes.append(child)


heapq.heapify(leave_nodes)
evicted_indices = []
evicted_size = 0

print_radix_tree(root_node)

# 删除空闲叶子结点，直到满足taget_size
while evicted_size < target_size:
    assert (
        leave_nodes
    ), f"Cannot evict enough cache, need {target_size}, only {evicted_size} evicted"
    node = heapq.heappop(leave_nodes)
    assert node.ref_count == 0 and node.is_leaf() and not node.is_root()
    evicted_size += node.length
    evicted_indices.append(node.slots)
    parent = node.parent
    del parent.children[int(node._token_ids[0].item())]
    print_radix_tree(root_node)
    print(f"Node: {node.uuid} is evicted")
    if parent.is_leaf() and parent.ref_count == 0:
        heapq.heappush(leave_nodes, parent)

free_slots = torch.cat(evicted_indices)

print()
print(f"free slots: {free_slots}")



from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, NamedTuple, Tuple, Dict
import time
import heapq


# 请求体定义：
@dataclass(eq=False)
class Request:
        uid: int
        input_ids: torch.Tensor  # cpu tensor
        table_idx: int
        cached_len: int
        output_len: int
        cache_handle: BaseCacheHandle =  None
        max_tokens: int = 1024

        @property
        def input_len(self) -> int:
            return len(self.input_ids)

# 计算整体尺寸大小：
class SizeInfo(NamedTuple):
    evictable_size: int
    protected_size: int
    @property
    def total_size(self) -> int:
        return self.evictable_size + self.protected_size


@dataclass(frozen=True)
class BaseCacheHandle(ABC):
    cached_len: int

@dataclass(frozen=True)
class RadixCacheHandle(BaseCacheHandle):
    node: RadixTreeNode

class BaseCacheManager(ABC):
    @abstractmethod
    def match_prefix(self, input_ids: torch.Tensor) -> Tuple[BaseCacheHandle, torch.Tensor]:
        """
        Match prefix and return the indices of the matched prefix in the cache.
        This operation will not modify the cache.
        The returned indices is only safe to use when the handle is locked.

        Args:
            input_ids (torch.Tensor): The input ids to match. Shape: (seq_len,)
        Returns:
            handle (BaseCacheHandle): The handle to the matched prefix.
            indices (torch.Tensor): The indices of the longest-matched prefix in the cache.
        """

    @abstractmethod
    def lock_handle(self, handle: BaseCacheHandle, unlock: bool = False) -> None:
        """
        Lock or unlock a cache handle.
        This operation will not modify the cache, but change the size info only.
        When a handle is locked, it cannot be evicted.
        Handles must be locked before the previously-returned tensor of `match_prefix` is used.
        Otherwise it may be evicted by calling evict.

        Args:
            handle (BaseCacheHandle): The cache handle to lock or unlock.
            unlock (bool): Whether to unlock the handle. Defaults to False.
        """

    @abstractmethod
    def insert_prefix(self, input_ids: torch.Tensor, indices: torch.Tensor) -> int:
        """
        Insert a new prefix into the cache.
        This operation will modify the cache.
        Args:
            input_ids (torch.Tensor): The input ids to insert. Shape: (seq_len,)
            indices (torch.Tensor): The indices to store the new prefix. Shape: (seq_len,)

        Returns:
            int: The length of prefix that is already in the cache. This part is not
                 inserted, so the caller should free these indices.
        """

    @abstractmethod
    def evict(self, size: int) -> torch.Tensor:
        """
        Evict some prefixes from the cache to free up space.
        This operation will modify the cache.
        Note that evict 0 is always safe and does nothing.
        Note that the actual evict size may be larger than the requested size.
        Args:
            size (int): The size to evict.

        Returns:
            torch.Tensor: The indices evicted. Shape: (evict_size,)
        Raises:
            RuntimeError: If the requested size is larger than the evictable size.
        """

    @abstractmethod
    def reset(self) -> None:
        """Reset the cache manager and the underlying cache."""

    @property
    @abstractmethod
    def size_info(self) -> SizeInfo:
        """Get the size information of the cache."""





class RadixCacheManager(BaseCacheManager):
    def __init__(self, device: torch.device):
        self.device = device
        self.empty_tensor = torch.empty(0, dtype=torch.int32, device=device)
        super().__init__()
        self.root_node = RadixTreeNode()
        self.root_node.ref_count = 1  # root is always protected
        self.evictable_size = 0
        self.protected_size = 0

    def lock_handle(self, handle: BaseCacheHandle, unlock: bool = False) -> None:
        if handle is None:
            return

        node = handle.node
        if unlock:
            while not node.is_root():
                node.ref_count -= 1
                assert node.ref_count >= 0
                if node.ref_count == 0:
                    self.evictable_size += node.length
                    self.protected_size -= node.length
                node = node.parent
        else:
            while not node.is_root():
                if node.ref_count == 0:
                    self.evictable_size -= node.length
                    self.protected_size += node.length
                node.ref_count += 1
                node = node.parent

    def match_prefix(self, input_ids: torch.Tensor) -> Tuple[RadixCacheHandle, torch.Tensor]:
        node, prefix_len = self._walk(input_ids)
        if prefix_len == 0:
            assert node.is_root() and node is self.root_node and prefix_len == 0
            return RadixCacheHandle(prefix_len, node), self.empty_tensor
        slots_list: List[torch.Tensor] = []
        matched_node = node
        while not node.is_root():
            slots_list.append(node.slots)
            node = node.parent
        slots_list.reverse()
        return RadixCacheHandle(prefix_len, matched_node), torch.cat(slots_list)

    def insert_prefix(self, input_ids: torch.Tensor, indices: torch.Tensor) -> int:
        node, prefix_len = self._walk(input_ids)
        assert prefix_len <= len(input_ids)
        if prefix_len < len(input_ids):
            new_node = RadixTreeNode()
            new_node.set_ids_slots(input_ids[prefix_len:], indices[prefix_len:].clone())
            new_node.set_parent(node)
            self.evictable_size += new_node.length
        return prefix_len

    def _walk(self, input_ids: torch.Tensor) -> Tuple[RadixTreeNode, int]:
        prefix_len = 0
        indice_len = len(input_ids)
        node = self.root_node
        tic = time.monotonic_ns()

        while prefix_len < indice_len:
            this_id = int(input_ids[prefix_len].item())
            if this_id not in node.children:
                return node, prefix_len

            node = node.children[this_id]

            # NOTE: at least 1 char is matched, so match_len >= 1
            match_len = node.get_match_len(input_ids[prefix_len:])
            prefix_len += match_len

            # need to split the node if not fully matched
            if match_len != node.length:
                node = node._split_at(match_len)
                return node, prefix_len

            # update timestamp for accessed node
            node.timestamp = tic

        return node, prefix_len

    def evict(self, size: int) -> torch.Tensor:
        if size == 0:
            return self.empty_tensor
        assert (
            size <= self.evictable_size
        ), f"Cannot evict {size}, only {self.evictable_size} is evictable"

        leave_nodes = self._collect_leave_nodes_for_evict()

        heapq.heapify(leave_nodes)
        evicted_indices: List[torch.Tensor] = []
        evicted_size = 0

        while evicted_size < size:
            assert (
                leave_nodes
            ), f"Cannot evict enough cache, need {size}, only {evicted_size} evicted"
            node = heapq.heappop(leave_nodes)
            assert node.ref_count == 0 and node.is_leaf() and not node.is_root()
            evicted_size += node.length
            evicted_indices.append(node.slots)
            self.evictable_size -= node.length
            parent = node.parent
            del parent.children[int(node._token_ids[0].item())]
            # NOTE: root is always protected, so won't be evicted
            if parent.is_leaf() and parent.ref_count == 0:
                heapq.heappush(leave_nodes, parent)

        return torch.cat(evicted_indices)

    def _collect_leave_nodes_for_evict(self) -> List[RadixTreeNode]:
        nodes: List[RadixTreeNode] = [self.root_node]
        leave_nodes: List[RadixTreeNode] = []

        while len(nodes) > 0:
            node = nodes.pop()
            if node.is_leaf():
                if node.ref_count == 0:
                    leave_nodes.append(node)
            else:
                for child in node.children.values():
                    nodes.append(child)

        return leave_nodes

    def reset(self) -> None:
        raise NotImplementedError("RadixManager.reset is not implemented")

    @property
    def size_info(self) -> SizeInfo:
        return SizeInfo(
            evictable_size=self.evictable_size,
            protected_size=self.protected_size,
        )



class CacheManager:
    def __init__(self, device: torch.device, num_pages: int):
        self._free_slots = torch.arange(num_pages, dtype=torch.int32, device=device)
        self.device = device
        self.manager = RadixCacheManager(device=device)
        self.num_pages = num_pages

    def _free(self, indices: torch.Tensor) -> None:
        if len(indices) > 0:
            self._free_slots = torch.cat([self._free_slots, indices])

    def match_req(self, req: Request):
        input_len = req.input_len
        assert input_len > 0, "Input length must be greater than 0."
        return self.manager.match_prefix(req.input_ids[: input_len - 1])

    @property
    def available_size(self) -> int:
        return self.manager.size_info.evictable_size + len(self._free_slots)

    def lock(self, handle: BaseCacheHandle) -> None:
        self.manager.lock_handle(handle, unlock=False)

    def unlock(self, handle: BaseCacheHandle) -> None:
        self.manager.lock_handle(handle, unlock=True)

    def allocate(self, needed_len: int) -> torch.Tensor:
        if needed_len <= (free_len := len(self._free_slots)):
            allocated = self._free_slots[:needed_len]
            self._free_slots = self._free_slots[needed_len:]
            return allocated

        # NOTE: len(evicted) + free_len >= needed_len
        evicted = self.manager.evict(needed_len - free_len)
        merged = torch.cat([self._free_slots, evicted])
        assert len(merged) >= needed_len, "Eviction did not free enough space."

        allocated = merged[:needed_len]
        self._free_slots = merged[needed_len:]
        return allocated

    def free_and_cache_finished_req(
        self,
        old_handle: BaseCacheHandle,
        input_ids: torch.Tensor,
        indices: torch.Tensor,
    ) -> None:
        in_cache_len = self.manager.insert_prefix(input_ids, indices)
        self._free(indices[old_handle.cached_len : in_cache_len])
        self.unlock(old_handle)




# https://zhuanlan.zhihu.com/p/1994495318197305400



