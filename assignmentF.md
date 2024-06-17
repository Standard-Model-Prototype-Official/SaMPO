# Assignment #F: All-Killed 满分

Updated 1844 GMT+8 May 20, 2024

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

### 22485: 升空的焰火，从侧面看

http://cs101.openjudge.cn/practice/22485/



思路：

bfs，但是只有一些会被记录

代码

```python
#2300012610ljx
from collections import deque

def right_view(n, tree):
    queue = deque([(1, tree[1])])
    right_view = []
    
    while queue:
        level_size = len(queue)
        for i in range(level_size):
            node, children = queue.popleft()
            
            if children[0] != -1:
                queue.append((children[0], tree[children[0]]))
            if children[1] != -1:
                queue.append((children[1], tree[children[1]]))
                
        right_view.append(node)
        
    return right_view

n = int(input())
tree = {1: [-1, -1] for _ in range(n+1)}
for i in range(1, n+1):
    left, right = map(int, input().split())
    tree[i] = [left, right]
    
result = right_view(n, tree)
print(*result) 

```



代码运行截图 ==（至少包含有"Accepted"）==

<img src="C:\Users\15744\AppData\Roaming\Typora\typora-user-images\image-20240528154652059.png" alt="image-20240528154652059" style="zoom:67%;" />



### 28203:【模板】单调栈

http://cs101.openjudge.cn/practice/28203/



思路：

单调栈

代码

```python
#2300012610ljx
n = int(input())
a = list(map(int, input().split()))
stack = []

for i in range(n):
    while stack and a[stack[-1]] < a[i]:
        a[stack.pop()] = i + 1
        
    stack.append(i)
    
while stack:
    a[stack[-1]] = 0
    stack.pop()
    
print(*a)

```



代码运行截图 ==（至少包含有"Accepted"）==

<img src="C:\Users\15744\AppData\Roaming\Typora\typora-user-images\image-20240528164519385.png" alt="image-20240528164519385" style="zoom:67%;" />



### 09202: 舰队、海域出击！

http://cs101.openjudge.cn/practice/09202/



思路：

dfs，图的环

代码

```python
#2300012610ljx
from collections import defaultdict

def dfs(node, color):
    color[node] = 1
    for neighbor in graph[node]:
        if color[neighbor] == 1:
            return True
        if color[neighbor] == 0 and dfs(neighbor, color):
            return True
    color[node] = 2
    return False

T = int(input())
for _ in range(T):
    N, M = map(int, input().split())
    graph = defaultdict(list)
    for _ in range(M):
        x, y = map(int, input().split())
        graph[x].append(y)
    color = [0] * (N + 1)
    is_cyclic = False
    for node in range(1, N + 1):
        if color[node] == 0:
            if dfs(node, color):
                is_cyclic = True
                break
    print('Yes' if is_cyclic else 'No')
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

<img src="C:\Users\15744\AppData\Roaming\Typora\typora-user-images\image-20240528170553906.png" alt="image-20240528170553906" style="zoom:67%;" />



### 04135: 月度开销

http://cs101.openjudge.cn/practice/04135/



思路：

bisect。和河中跳房子有点像，但是非常难想到

代码

```python
#2300012610ljx
n,m = map(int, input().split())
expenditure = []
for _ in range(n):
    expenditure.append(int(input()))

def check(x):
    num, s = 1, 0
    for i in range(n):
        if s + expenditure[i] > x:
            s = expenditure[i]
            num += 1
        else:
            s += expenditure[i]
    
    return num > m

lo = max(expenditure)
hi = sum(expenditure) + 1
ans = 1
while lo < hi:
    mid = (lo + hi) // 2
    if check(mid):      
        lo = mid + 1
    else:
        ans = mid    
        hi = mid
        
print(ans)
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

<img src="C:\Users\15744\AppData\Roaming\Typora\typora-user-images\image-20240528172208019.png" alt="image-20240528172208019" style="zoom:67%;" />



### 07735: 道路

http://cs101.openjudge.cn/practice/07735/



思路：

dijkstra

代码

```python
#2300012610ljx
import heapq

def dijkstra(g):
    while pq:
        dist, node, fee = heapq.heappop(pq)
        if node == n - 1:
            return dist
        for nei, w, f in g[node]:
            n_dist = dist + w
            n_fee = fee + f
            if n_fee <= k:
                dists[nei] = n_dist
                heapq.heappush(pq, (n_dist, nei, n_fee))
    return -1

k, n, r = int(input()), int(input()), int(input())
g = [[] for _ in range(n)]
for i in range(r):
    s, d, l, t = map(int, input().split())
    g[s - 1].append((d - 1, l, t))
    
pq = [(0, 0, 0)]
dists = [float('inf')] * n
dists[0] = 0
spend = 0

result = dijkstra(g)
print(result)

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

<img src="C:\Users\15744\AppData\Roaming\Typora\typora-user-images\image-20240528173951362.png" alt="image-20240528173951362" style="zoom:67%;" />



### 01182: 食物链

http://cs101.openjudge.cn/practice/01182/



思路：

并查集

代码

```python
#2300012610ljx
def find(x):
    if p[x] == x:
        return x
    else:
        p[x] = find(p[x])
        return p[x]
    
n, k = map(int, input().split())

p = [0] * (3 * n + 1)
for i in range(3 * n + 1):
    p[i] = i
    
ans = 0
for _ in range(k):
    a, x, y = map(int, input().split())
    if x > n or y > n:
        ans += 1
        continue
    
    if a == 1:
        if find(x + n) == find(y) or find(y + n) == find(x):
            ans += 1
            continue
        
        p[find(x)] = find(y)
        p[find(x + n)] = find(y + n)
        p[find(x + 2 * n)] = find(y + 2 * n)
    else:
        if find(x) == find(y) or find(y + n) == find(x):
            ans += 1
            continue
            
        p[find(x + n)] = find(y)
        p[find(y + 2 * n)] = find(x)
        p[find(x + 2 * n)] = find(y + n)
        
print(ans)

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

<img src="C:\Users\15744\AppData\Roaming\Typora\typora-user-images\image-20240528182711994.png" alt="image-20240528182711994" style="zoom:67%;" />



## 2. 学习总结和收获

==如果作业题目简单，有否额外练习题目，比如：OJ“2024spring每日选做”、CF、LeetCode、洛谷等网站题目。==

《荀子·儒效》：“……与时迁徙，与世偃仰，千举万变，其道一也……”

作业还是挺难的，但是也都是从学过的地方出发，各种变化。思路上最难的应该是月度开销，实现上最难的应该是食物链。希望机考能有好成绩

下面是让ai写的单调栈和单调队列模板

### 单调栈

```python
def find_left_greater(nums):
    stack = []
    result = [-1] * len(nums)

    for i in range(len(nums)):
        while stack and nums[stack[-1]] < nums[i]:
            result[stack.pop()] = nums[i]
        stack.append(i)

    return result

# 使用示例
nums = [2, 1, 4, 3, 5]
print(find_left_greater(nums))  # 输出: [-1, -1, 2, 2, 4]
```

### 单调队列

```python
from collections import deque

def max_in_window(nums, k):
    queue = deque()
    result = []

    for i in range(len(nums)):
        # 将队列中比当前元素小的都弹出
        while queue and nums[queue[-1]] < nums[i]:
            queue.pop()
        queue.append(i)

        # 如果窗口左边界已经不在队列中了,则将其从队列中移除
        if queue[0] == i - k:
            queue.popleft()

        # 如果窗口大小达到k,则将队列头部元素(即窗口内最大值)加入结果
        if i >= k - 1:
            result.append(nums[queue[0]])

    return result

# 使用示例
nums = [1, 3, -1, -3, 5, 3, 6, 7]
print(max_in_window(nums, 3))  # 输出: [3, 3, 5, 5, 6, 7]
```

