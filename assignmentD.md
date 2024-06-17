# Assignment #D: May月考

Updated 1654 GMT+8 May 8, 2024

2024 spring, Complied by ==同学的姓名、院系==



**说明：**

1）请把每个题目解题思路（可选），源码Python, 或者C++（已经在Codeforces/Openjudge上AC），截图（包含Accepted），填写到下面作业模版中（推荐使用 typora https://typoraio.cn ，或者用word）。AC 或者没有AC，都请标上每个题目大致花费时间。

2）提交时候先提交pdf文件，再把md或者doc文件上传到右侧“作业评论”。Canvas需要有同学清晰头像、提交文件有pdf、"作业评论"区有上传的md或者doc附件。

3）如果不能在截止前提交作业，请写明原因。



**编程环境**

==（请改为同学的操作系统、编程环境等）==

操作系统：macOS Ventura 13.4.1 (c)

Python编程环境：Spyder IDE 5.2.2, PyCharm 2023.1.4 (Professional Edition)

C/C++编程环境：Mac terminal vi (version 9.0.1424), g++/gcc (Apple clang version 14.0.3, clang-1403.0.22.14.1)



## 1. 题目

### 02808: 校门外的树

http://cs101.openjudge.cn/practice/02808/



思路：

找到了自己去年写的代码。思路大同小异，用数组模拟实现这个过程

代码

```python
#2300012610ljx
L, M = map(int, input().split())
L += 1
array = [1]* L
for i in range(M):
    start, end = map(int, input().split())
    end += 1
    for m in range(start, end):
        array[m] = 0
print(sum(array))
```



代码运行截图 ==（至少包含有"Accepted"）==

<img src="C:\Users\15744\AppData\Roaming\Typora\typora-user-images\image-20240521145803853.png" alt="image-20240521145803853" style="zoom:67%;" />



### 20449: 是否被5整除

http://cs101.openjudge.cn/practice/20449/



思路：

秦九韶算法

代码

```python
#2300012610ljx
nmbr = input()
nswr = ''
sprt = 0
for char in nmbr:
    sprt <<= 1
    sprt += int(char)
    if sprt% 5 == 0:
        nswr += '1'
    else:
        nswr += '0'
print(nswr)
```



代码运行截图 ==（至少包含有"Accepted"）==

<img src="C:\Users\15744\AppData\Roaming\Typora\typora-user-images\image-20240521150732122.png" alt="image-20240521150732122" style="zoom:67%;" />



### 01258: Agri-Net

http://cs101.openjudge.cn/practice/01258/



思路：

最小生成树，稠密用prim

代码

```python
#2300012610ljx
from heapq import heappop, heappush, heapify

def prim(graph, start_node):
    mst = set()
    visited = set([start_node])
    edges = [
        (cost, start_node, to)
        for to, cost in graph[start_node].items()
    ]
    heapify(edges)
    
    while edges:
        cost, frm, to = heappop(edges)
        if to not in visited:
            visited.add(to)
            mst.add((frm, to, cost))
            for to_next, cost2 in graph[to].items():
                if to_next not in visited:
                    heappush(edges, (cost2, to, to_next))
                    
    return mst

while True:
    try:
        N = int(input())
    except EOFError:
        break
    
    graph = {i: {} for i in range(N)}
    for i in range(N):
        for j, cost in enumerate(map(int, input().split())):
            graph[i][j] = cost
            
    mst = prim(graph, 0)
    total_cost = sum(cost for frm, to, cost in mst)
    print(total_cost)
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

<img src="C:\Users\15744\AppData\Roaming\Typora\typora-user-images\image-20240521170810091.png" alt="image-20240521170810091" style="zoom:67%;" />



### 27635: 判断无向图是否连通有无回路(同23163)

http://cs101.openjudge.cn/practice/27635/



思路：

总觉得做过了，但是翻一下又没有。思路基本上是dfs+disjointset

代码

```python
#2300012610ljx
def is_connected(graph, n):
    visited = [False] * n
    stack = [0]
    visited[0] = True
    
    while stack:
        node = stack.pop()
        for neighbor in graph[node]:
            if not visited[neighbor]:
                stack.append(neighbor)
                visited[neighbor]= True
                
    return all(visited)

def has_cycle(graph, n):
    def dfs(node, visited, parent):
        visited[node] = True
        for neighbor in graph[node]:
            if not visited[neighbor]:
                if dfs(neighbor, visited, node):
                    return True
            elif parent != neighbor:
                return True
        return False
    
    visited = [False] * n
    for node in range(n):
        if not visited[node]:
            if dfs(node, visited, -1):
                return True
    return False

n, m = map(int, input().split())
graph = [[] for _ in range(n)]
for _ in range(m):
    u, v = map(int, input().split())
    graph[u].append(v)
    graph[v].append(u)
    
print('connected:yes' if is_connected(graph, n) else 'connected:no')
print('loop:yes' if has_cycle(graph, n) else 'loop:no')

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

<img src="C:\Users\15744\AppData\Roaming\Typora\typora-user-images\image-20240521180545847.png" alt="image-20240521180545847" style="zoom:67%;" />





### 27947: 动态中位数

http://cs101.openjudge.cn/practice/27947/



思路：

用两个堆来回倒

代码

```python
#2300012610ljx
import heapq

def dynamic_median(nums):
    min_heap = []
    max_heap = []
    
    median = []
    for i, num in enumerate(nums):
        if not max_heap or num <= -max_heap[0]:
            heapq.heappush(max_heap, -num)
        else:
            heapq.heappush(min_heap, num)
            
        if len(max_heap) - len(min_heap) > 1:
            heapq.heappush(min_heap, -heapq.heappop(max_heap))
        elif len(min_heap) > len(max_heap):
            heapq.heappush(max_heap, -heapq.heappop(min_heap))
            
        if i % 2 == 0:
            median.append(-max_heap[0])
            
    return median


T = int(input())
for _ in range(T):
    nums = list(map(int, input().split()))
    median = dynamic_median(nums)
    print(len(median))
    print(*median)
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

<img src="C:\Users\15744\AppData\Roaming\Typora\typora-user-images\image-20240521200021618.png" alt="image-20240521200021618" style="zoom:67%;" />



### 28190: 奶牛排队

http://cs101.openjudge.cn/practice/28190/



思路：

单调栈

代码

```python
#2300012610ljx
N = int(input())
heights = [int(input()) for _ in range(N)]

left_bound = [-1] * N
right_bound = [N] * N

stack = []

for i in range(N):
    while stack and heights[stack[-1]] < heights[i]:
        stack.pop()
        
    if stack:
        left_bound[i] = stack[-1]
        
    stack.append(i)
    
for i in range(N-1, -1, -1):
    while stack and heights[stack[-1]] > heights[i]:
        stack.pop()
        
    if stack:
        right_bound[i] = stack[-1]
        
    stack.append(i)
    
ans = 0

for i in range(N):
    for j in range(left_bound[i] + 1, i):
        if right_bound[j] > i:
            ans = max(ans, i - j + 1)
            break
print(ans)
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

<img src="C:\Users\15744\AppData\Roaming\Typora\typora-user-images\image-20240521204129118.png" alt="image-20240521204129118" style="zoom:67%;" />



## 2. 学习总结和收获

==如果作业题目简单，有否额外练习题目，比如：OJ“2024spring每日选做”、CF、LeetCode、洛谷等网站题目。==

大部分都是复习了，但是也有一些题目完全可以用计概办法完成。感觉看个人理解

最难的题，应该是动态中位数的二重堆和奶牛的单调栈。



