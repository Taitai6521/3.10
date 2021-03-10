# import requests
# class ThirdPartyBonusRestApi(object):
#     def bonus_price(self, year):
#         r = requests.get('http://localhost/bonus', params={'year': year})
#         return r.json()['price']
# class Salary(object):
#     def __init__(self, base=100, year=2017):
#         self.bonus_api = ThirdPartyBonusRestApi()
#         self.base = base
#         self.year = year
#     def calculation_salary(self):
#         bonus = self.bonus_api.bonus_price(year=self.year)
#         return self.base + bonus
#
#
# import logging
# import threading
# import time
#
#
# logging.basicConfig(
#     level = logging.DEBUG, format='%(threadName)s: %(message)s')
#
# def worker1(x, y=1):
#     logging.debug('start')
#     logging.debug(x)
#     logging.debug(y)
#     time.sleep(5)
#     logging.debug('end')
#
#
# def worker2():
#     logging.debug('start')
#     time.sleep(2)
#     logging.debug('end')
#
#
# # if __name__ == '__main__':
# #
# #     threads = []
# #     for _ in range(5):
# #
# #
# #
# #         t1 = threading.Thread(target=worker1)
# #         t1.setDaemon(True)
# #         t1.start()
# #         # threads.append(t)
# #     print(threading.enumerate())
# #     for thread in threading.enumerate():
# #         if thread is threading.currentThread():
# #             print(thread)
# #             continue
# #         thread.join()
# if __name__ == '__main__':
#     t = threading.Timer(3, worker1, args=(100,), kwargs={'y': 100})
#     t.start()
#
#     # t2 = threading.Thread(target=worker2)
#     #
#     # t1.start()
#     # t2.start()
#     # print('started')
#     # t1.join()
#     #






# from typing import List
#
# def insertion_sort(numbers: List[int]) -> List[int]:
#     len_numbers = len(numbers)
#     for i in range(1, len_numbers):
#         temp = numbers[i]
#         j = i - 1
#
#         while j >= 0 and numbers[j] > temp:
#             numbers[j+1] = numbers[j]
#             j -= 1
#
#         numbers[j+1] = temp
#     return numbers
#
# def bucket_sort(numbers: List[int]) -> List[int]:
#     max_nums = max(numbers)
#     len_numbers = len(numbers)
#
#     size = max_num // len_numbers
#
#     buckets = [[] for _ in range(size)]
#     for num in numbers:
#
#         i = num // size
#         if i != size:
#             buckets[i].append(num)
#
#         else:
#             buckets[size-1].append(num)
#     for i in range(size):
#         insertion_sort(buckets[i])
#     result = []
#     for i in range(size):
#         result += buckets
#     return result
#
#
# if __name__ == '__main__':
#
#     nums = [1,4,4,32,24,5]
#     print(insertion_sort(nums))


# from typing import List
#
#
# def shell_sort(numbers: List[int]) -> List[int]:
#
#     len_numbers = len(numbers)
#
#     gap = len_numbers // 2
#     while gap > 0:
#         for i in range(gap, len_numbers):
#             temp = numbers[i]
#             j = i
#             while j >= gap and numbers[j-gap] > temp:
#                 numbers[j] = numbers[j-gap]
#                 j -= 1
#                 numbers[j] = temp
#             gap //= 2
#         return numbers
#
#
#
# if __name__ == '__main__':

#
# from typing import List
#
# def counting_sort(numbers: List[int], place: int) -> List[int]:
#     counts = [0] * 10
#     result = [0] * len(numbers)
#     for num in numbers:
#         index = int(num / place) % 10
#         counts[index] += 1
#
#     for i in range(1, 10):
#         counts[i] += counts[i-1]
#     i = len(numbers) -1
#
#     while i >= 0:
#         index = numbers[i]
#         result[counts[index] - 1] = numbers[i]
#         counts[index] -= 1
#         i -= 1
#     return result
#
#
#
# def radix_sort(numbers: List[int]) -> List[int]:x(numbers)
#     place = 1
#     while max_num > place:
#         numbers = counting_sort(numbers, place)
#         place *= 10
#     return numbers
#
#
#
#
#
# if __name__ == '__main__':
#
#     nums = [13,3,5,45,5]
#     print(counting_sort(nums))
#



#
# from typing import List
#
#
# def partition(numbers: List[int], low: int, high: int, ) -> int:
#
#     i = low - 1
#     pivot = numbers[high]
#     for j in range(low, high):
#         if numbers[j] <= pivot:
#             i += 1
#             numbers[i], numbers[j] = numbers[j], numbers[i]
#     numbers[i+1], numbers[high] = numbers[high], numbers[i+1]
#     return i+1
#
#
#
#
# def quick_sort(numbers: List[int]) -> List[int]:
#     def _quick_sort(numbers: List[int], low: int, high: int) -> None:
#         if low < high:
#             partition_index = partition(numbers, low, high)
#             partition_index = partition(numbers, low, partition_index-1)
#             partition_index = partition(numbers, partition_index+1, high)
#
#     _quick_sort(numbers, 0, len(numbers) -1)
#     return numbers
#
#
#
# if __name__ == '__main__':
#     nums = [1,3,4,8,4,4]
#     print(quick_sort(nums))


#
# from typing import List
#
#
# def merge_sort(numbers: List[int]) -> List[int]:
#     if len(numbers) <= 1:
#         return numbers
#
#     center = len(numbers) // 2
#     left = numbers[:center]
#     right = numbers[center:]
#
#     merge_sort(left)
#     merge_sort(right)
#
#
#     i = j = k = 0
#     while i < len(left) and j < len(right):
#         if left[i] <= right[j]:
#             numbers[k] = left[i]
#             i += 1
#         else:
#             numbers[k] = right[j]
#             j += 1
#         k += 1
#
#
#     while i < len(left):
#         numbers[k] = left[i]
#         i += 1
#         k += 1
#
#     while j < len(right):
#         numbers[k] = right[j]
#         j += 1
#         k += 1
#
#     return numbers
#
# if __name__ == '__main__':
#
#
#     nums = [1,4,4,44,5]
#     print(merge_sort(nums))

import sys
from typing import Optional

#
# class MiniHeap(object):
#
#     def __init__(self) -> None:
#         self.heap = [-1 * sys.maxsize]
#         self.current_size = 0
#
#     def parent_index(self, index: int) -> int:
#         return index // 2
#
#     def left_child_index(self, index: int) -> int:
#         return 2 * index
#
#     def right_child_index(self, index: int) -> int:
#         return (2 * index) + 1
#
#     def swap(self, index1: int, index2: int) -> None:
#         self.heap[index1], self.heap[index2] = self.heap[index2], self.heap[index1]
#
#     def heapify_up(self, index: int) -> None:
#         while self.parent_index(index) > 0:
#             if self.heap[index] < self.heap[self.parent_index(index)]:
#                 self.swap(index, self.parent_index(index))
#             index = self.parent_index(index)
#
#     def push(self, value: int) -> None:
#         self.heap.append(value)
#         self.current_size += 1
#         self.heapify_up(self.current_size)
#
#     def min_child_index(self, index: int) -> int:
#         if self.right_child_index(index) > self.current_size:
#             return self.left_child_index(index)
#         else:
#             if (self.heap[self.left_child_index(index)] <
#                 self.heap[self.right_child_index(index)]):
#                 return self.left_child_index(index)
#             else:
#                 return self.right_child_index(index)
#
#     def heapify_down(self, index: int) -> None:
#         while self.left_child_index(index) <= self.current_size:
#             min_child_index = self.min_child_index(index)
#             if self.heap[index] > self.heap[min_child_index]:
#                 self.swap(index, min_child_index)
#             index = min_child_index
#
#     def pop(self) -> Optional[int]:
#         if len(self.heap) == 1:
#             return
#
#         root = self.heap[1]
#         data = self.heap.pop()
#         if len(self.heap) == 1:
#             return root
#
#         # [-x, 5, 6, 2, 9, 13, 11]
#         self.heap[1] = data
#         self.current_size -= 1
#         self.heapify_down(1)
#         return root
#
#
#
#
# if __name__ == '__main__':
#     min_heap = MiniHeap()
#     min_heap.push(5)
#     min_heap.push(6)
#     min_heap.push(2)
#     min_heap.push(9)
#     min_heap.push(13)
#     min_heap.push(11)
#     min_heap.push(1)
#     print(min_heap.heap)
#     print(min_heap.pop())
#     print(min_heap.heap)

#
# def merge_sort(data: list, l: int, m: int, r: int) -> list:
#     len_left, len_right = m - l + 1, r - m
#     left, right = [], []
#     for i in range(0, len_left):
#         left.append(data[l + i])
#     for i in range(0, len_right):
#         right.append(data[m + 1 + i])
#
#     i, j, k = 0, 0, l
#     while i < len_left and j < len_right:
#         if left[i] <= right[j]:
#             data[k] = left[i]
#             i += 1
#         else:
#             data[k] = right[j]
#             j += 1
#         k += 1
#
#     while i < len_left:
#         data[k] = left[i]
#         k += 1
#         i += 1
#
#     while j < len_right:
#         data[k] = right[j]
#         k += 1
#         j += 1
#
#     return data
#
#
# def insertion_sort(data: list, left: int, right: int) -> list:
#     for i in range(left + 1, right + 1):
#         temp = data[i]
#         j = i - 1
#         while j >= left and data[j] > temp:
#             data[j + 1] = data[j]
#             j -= 1
#
#         data[j + 1] = temp
#     return data
#
#
# def tim_sort(data: list, size: int = 32) -> list:
#     n = len(data)
#     for i in range(0, n, size):
#         insertion_sort(data, i, min((i + 31), (n - 1)))
#
#     while size < n:
#         for left in range(0, n, 2 * size):
#             mid = left + size - 1
#             right = min((left + 2 * size - 1), (n - 1))
#             merge_sort(data, left, mid, right)
#         size = 2 * size
#     return data
#
#
# if __name__ == '__main__':
#     import random
#     nums = [random.randint(0, 1000) for _ in range(1000)]
#     print(tim_sort(nums))



#
# from typing import List
#
#
# def bubble_sort(numbers: List[int]) -> List[int]:
#     len_numbers = len(numbers)
#     for i in range(len_numbers):
#         for j in range(len_numbers - 1 - i):
#             if numbers[j] > numbers[j+1]:
#                 numbers[j], numbers[j+1] = numbers[j+1], numbers[j]
#     return numbers
#
#
#
#         print(i)
# if __name__ == '__main__':
#     nums = [2,3,334,4,4]
#     bubble_sort(nums)
#




# from typing import List
#
#
# def selection_sort(numbers: List[int]) -> List[int]:
#     len_numbers = len(numbers)
#     for i in range(len_numbers):
#         min_idx = i
#         for j in range(i+1, len_numbers):
#             if numbers[min_idx] > numbers[j]:
#                 min_idx = j
#
#         numbers[i], numbers[min_idx] = numbers[min_idx], numbers[i]
#
#     return numbers
#
# if __name__ == '__main__':
#     nums = [14,4,4,45]
#     print(selection_sort(nums))
#
#

#
# from typing import List
#
# def selection_sort(numbers: List[int]) -> List[int]:
#     len_numbers = len(numbers)
#     for i in range(len_numbers):
#         min_idx = i
#         for j in range(i+1, len_numbers):
#             if numbers[min_idx] > numbers[j]:
#                 min_idx = j
#         numbers[i], numbers[min_idx] = numbers[min_idx], numbers[i]
#     return numbers
#
#
# if __name__ == '__main__':
#     nums = [34,4,55,6,34]
#     print(selection_sort(nums))




#
# from typing import List, Iterator, Tuple
#
#
# def find_pair(pairs: List[Tuple[int, int]]) -> Iterator[Tuple[int, int]]:
#     cache = {}
#     for pair in pairs:
#         first, second = pair[0], pair[1]
#
#         value = cache.get(second)
#         if not value:
#             cache[first] = second
#         elif value == first:
#             yield pair
#
#
# if __name__ == '__main__':
#     l = [(1,3),(3, 1)]
#
#     for r in find_pair(l):
#         print(r)
#


#
# import operator
# from typing import Tuple
#
# from collections import Counter
#
# def count_chars_v1(string: str) -> Tuple[str, int]:
#     strings = string.lower()
#     # l = []
#     # for char in strings:
#     #     if not char.isspace():
#     #        l.append((char, strings.count(char)))
#     #
#     l = [(c, strings.count(c)) for c in strings if not c.isspace()]
#
#     return max(l, key=operator.itemgetter(1))
#
#
# def count_chars_v2(string: str) -> Tuple[str, int]:
#     strings = string.lower()
#     d = {}
#     for char in strings:
#         if not char.isspace():
#             d[char] = d.get(char, 0) + 1
#     max_key = max(d,key=d.get)
#     return max_key, k[max_key]
#
#
# def count_chars_v3(string: str) -> Tuple[str, int]:
#     strings = string.lower()
#     d = Counter()
#     for char in strings:
#         if not char.isspace():
#             d[char] += 1
#     max_key = max(d,key=d.get)
#     return max_key, k[max_key]
#
# if __name__ == '__main__':
#     s = 'This is a pen. This is a apple'
#     print(count_chars_v1(s))


from functools import lru_cache
# import time
#
# def memoize(f):
#     def _wrapper(n):
#         if n not in cache:
#
#             cache[n] = f(n)
#         return r
#     return _wrapper()
#
#
#
# @memoize
# def long_func(num: int) -> int:
#     r = 0
#     for i in range(1000):
#         r += num*i
#     return r
#
#
#
#
#
#
#
# if __name__ == '__main__':
#     for i in range(10):
#         print(long_func(10))
#     start = time.time()
#
#     for i in range(10):
#         print(long_func(i))
#
#     print(time.time() - start)
#


#
#
# from typing import List
# from collections import Counter
#
#
# def min_count_remove(x: List[int], y: List[int]) -> None:
# #     count_x = {}
# #     count_y = {}
# #     for i in x:
# #         count_x[i] = count_x.get(i, 0) + 1
# #
# #     for i in x:
# #         count_y[i] = count_y.get(i, 0) + 1
# #     print(count_x)
# #     print(count_y)
#     counter_x = Counter(x)
#     counter_y = Counter(y)
#     print(counter_x)
#     print(counter_y)
#
#     for key_x, value_x in counter_x.items():
#         value_y = counter_y.get(key_x)
#         if value_y:
#             if value_x < value_y:
#                 x[:] = [i for i in x if i != key_x]
#             elif value_x > value_y:
#                 y[:] = [i for i in y if i != key_x]
#
#
#
# if __name__ == '__main__':
#     x = [1,3,3,3,44,55,5]
#     y = [2,3,4,44,5,8,5]
#     min_count_remove(x, y)
#     print(x)
#     print(y)


# l = [1,2,3]
# print(int(''.join([str(i) for i in l])) + 1)
# from typing import List
#
#
#
# def remove_zero(numbers: List[int]) -> None:
#     if numbers and numbers[0] == 0:
#         numbers.pop(0)
#         remove_zero(numbers)
#
#
# def list_to_into(numbers: List[int]) -> List[int]:
#     sum_numbers = 0
#     for i, num in enumerate(reversed(numbers)):
#         sum_numbers += num * (10**i)
#     return sum_numbers
# def list_to_int_plus_one(numbers: List[int]) -> int:
#     i = len(numbers) - 1
#
#     numbers[i] += 1
#     while 0 < i:
#
#         if numbers[i] != 10:
#             remove_zero(numbers)
#             break
#         numbers[i] = 0
#         numbers[i-1] += 1
#
#         i -= 1
#
#     else:
#         if numbers[0] == 10:
#             numbers[0] = 1
#             numbers.append(0)
#
#     return list_to_into(numbers)
#
#
#
# if __name__ == '__main__':
#     print(list_to_int_plus_one([9,9,9]))






#
# from typing import List
#
# def snake_string_v1(chars: str) -> List[List[int]]:
#     result = [[], [], []]
#     result_index = {0, 1, 2}
#     insert_index = 1
#     for i, s in enumerate(chars):
#         if i % 4 == 1:
#             insert_index = 0
#         elif i % 2 == 0:
#             insert_index = 1
#         elif i % 4 == 3:
#             insert_index = 2
#             result[insert_index].append(s)
#             for rest_index in result_index - {insert_index}:
#                 result[rest_index].append(' ')
#     return result
#
# if __name__ == '__main__':
#     for line in snake_string_v1('01234'):
#         print(''.join(line))
#

#
# import logging
# import queue
# import threading
# import time
#
# logging.basicConfig(
#     level=logging.DEBUG, format='%(threadName)s: %(message)s')
#
#
# def worker1(queue):
#         logging.debug('start')
#         while True:
#             item = queue.get()
#             if item is None:
#                 break
#             logging.debug(item)
#             queue.task_done()
#
#         logging.debug('end')
# #
# # def worker2(queue):
# #         logging.debug('start')
# #         time.sleep(5)
# #         print(queue.get())
# #         print(queue.get())
# #         logging.debug('end')
#
# if __name__ == '__main__':
#     queue = queue.Queue()
#     for i in range(10):
#         queue.put(i)
#     ts = []
#     for _ in range(3):
#
#         t1 = threading.Thread(target=worker1, args=(queue,))
#     # t2 = threading.Thread(target=worker2, args=(queue,))
#
#
#
#
#     t1.start()
#
#     logging.debug('task are not done')
#     queue.join()
#     # t2.start()
#     logging.debug('task are not done')
#     for _ in range(len(ts)):
#         queue.put(None)
#         t1.join()
#
#     for t in range(len(ts)):
#         queue.put(None)
#
#     [t.join() for i in ts]


import logging
import queue
import _multiprocessing
import threading
import time


logging.basicConfig(
    level=logging.DEBUG, format='%(processName)s, %(message)s')

def worker1(i):
        logging.debug('start')
        logging.debug('i')
        time.sleep(5)
        logging.debug('end')
        return i
if __name__ == '__main__':
    i = 10
    with _multiprocessing.Pool(3) as p:
        logging.debug(p.apply(worker1, (200,)))
        p1.apply_async(worker1, (100, ))
        p2.apply_async(worker1, (100, ))
        logging.debug('excuted')
        logging.debug(p1.get(timeout=1))
        logging.debug(p2.get())