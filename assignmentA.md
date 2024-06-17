# Assignment #A: 图论：算法，树算及栈

Updated 2018 GMT+8 Apr 21, 2024

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

### 20743: 整人的提词本

http://cs101.openjudge.cn/practice/20743/



思路：

用栈逆转！

代码

```python
#2300012610ljx
def reverse(s):
    stack = []
    for char in s:
        if char == ')':
            temp = []
            while stack and stack[-1] != '(':
                temp.append(stack.pop())
                
            if stack:
                stack.pop()
            
            stack.extend(temp)
        else:
            stack.append(char)
    return ''.join(stack)

s = input().strip()
print(reverse(s))

```



代码运行截图 ==（至少包含有"Accepted"）==

<img src="C:\Users\15744\AppData\Roaming\Typora\typora-user-images\image-20240507235653575.png" alt="image-20240507235653575" style="zoom:50%;" />



### 02255: 重建二叉树

http://cs101.openjudge.cn/practice/02255/



思路：

做了很多次了

代码

```python
def build_tree(preorder, inorder):
    if not preorder:
        return ''
    
    root = preorder[0]
    root_index = inorder.index(root)
    
    left_preorder = preorder[1:1 + root_index]
    right_preorder = preorder[1 + root_index:]
    
    left_inorder = inorder[:root_index]
    right_inorder = inorder[root_index + 1:]
    
    left_tree = build_tree(left_preorder, left_inorder)
    right_tree = build_tree(right_preorder, right_inorder)
    
    return left_tree + right_tree + root

while True:
    try:
        preorder, inorder = input().split()
        postorder = build_tree(preorder, inorder)
        print(postorder)
    except EOFError:
        break

```



代码运行截图 ==（至少包含有"Accepted"）==

<img src="C:\Users\15744\AppData\Roaming\Typora\typora-user-images\image-20240508000233321.png" alt="image-20240508000233321" style="zoom:67%;" />



### 01426: Find The Multiple

http://cs101.openjudge.cn/practice/01426/

要求用bfs实现



思路：

bfs, 检验模. 用队列辅助

代码

```python
#2300012610ljx
from collections import deque

def find_multiple(n):
    q = deque()
    q.append((i % n, "1"))
    visited = set([1 % n])
    
    while q:
        mod, num_str = q.popleft()
        
        if mod == 0:
            return num_str
        
        for digit in ['0', '1']:
            new_num_str = num_str + digit
            new_mod = (mod * 10 + int(digit)) % n
            
            if new_mod not in visited:
                q.append((new_mod, new_num_str))
                visited.add(new_mod)
                
def main():
    while True:
        n = int(input())
        if n == 0:
            break
        print(find_multiple(n))
        
if __name__ == "__main__":
    main()

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

<img src="C:\Users\15744\AppData\Roaming\Typora\typora-user-images\image-20240508153418167.png" alt="image-20240508153418167" style="zoom:67%;" />



### 04115: 鸣人和佐助

bfs, http://cs101.openjudge.cn/practice/04115/



思路：

有特殊判定条件的bfs

代码

```python
#2300012610ljx
from collections import deque

M, N, T = map(int, input().split())
graph = [list(input()) for i in range(M)]
direc = [(0, 1), (1, 0), (-1, 0), (0, -1)]
start, end = None, None
for i in range(M):
    for j in range(N):
        if graph[i][j] == '@':
            start = (i, j)
            
def bfs():
    q = deque([start + (T, 0)])
    visited = [[-1] * N for _ in range(M)]
    visited[start[0]][start[1]] = T
    while q:
        x, y, t, time = q.popleft()
        time += 1
        for dx, dy in direc:
            nx, ny = x+dx, y+dy
            if 0<=nx<M and 0<=ny<N:
                if (elem := graph[nx][ny]) == '*' \
                    and t > visited[nx][ny]:
                    visited[nx][ny] = t
                    q.append((nx, ny, t, time))
                elif elem == '#' and t > 0\
                    and t-1 > visited[nx][ny]:
                    visited[nx][ny] = t-1
                    q.append((nx, ny, t-1, time))
                elif elem == '+':
                    return time
    return -1

print(bfs())
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

<img src="C:\Users\15744\AppData\Roaming\Typora\typora-user-images\image-20240508155623369.png" alt="image-20240508155623369" style="zoom:67%;" />



### 20106: 走山路

Dijkstra, http://cs101.openjudge.cn/practice/20106/



思路：

dijkstra = bfs + prior queue. 找到了自己半年前的汉字代码，没绷住，我竟然还能看懂。

代码

```python
#2300012610ljx
import heapq
def 索(甲, 乙, 丙, 丁):
    变 = [[-1,0],[1,0],[0,1],[0,-1]]
    队 = []
    heapq.heappush(队, [0, 甲, 乙])
    径 = set((甲, 乙))
    if 阵[甲][乙] == "#" or 阵[丙][丁] == "#":
        return "NO"
    else:
        while 队:
            步, 横, 纵 = map(int, heapq.heappop(队))
            径.add((横, 纵))
            if 横 == 丙 and 纵 == 丁:
                return 步
            for 无 in 变:
                新横, 新纵 = 横 + 无[0], 纵 + 无[1]
                if -1 < 新横 < 宽 and -1 < 新纵 < 长:
                    if 阵[新横][新纵] != "#" and (新横, 新纵) not in 径:
                        高差 = abs(int(阵[横][纵])-int(阵[新横][新纵]))
                        heapq.heappush(队, [步 + 高差, 新横, 新纵])
        return "NO"

阵 = []         
数据集 = []   
宽, 长, 数 = map(int, input().split())
for 甲 in range(宽):
    阵.append(input().split())
for 乙 in range(数):
    数据集.append(list(map(int, input().split())))
for 丙 in 数据集:
    print(索(*丙))
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

<img src="C:\Users\15744\AppData\Roaming\Typora\typora-user-images\image-20240508160740268.png" alt="image-20240508160740268" style="zoom:67%;" />



### 05442: 兔子与星空

Prim, http://cs101.openjudge.cn/practice/05442/



思路：

prim = greedy + tree

代码

```python
#2300012610ljx
import heapq

def prim(graph, start):
    mst = []
    used = set([start])
    edges = [
        (cost, start, to)
        for to, cost in graph[start].items()
    ]
    heapq.heapify(edges)
    
    while edges:
        cost, frm, to = heapq.heappop(edges)
        if to not in used:
            used.add(to)
            mst.append((frm, to, cost))
            for to_next, cost2 in graph[to].items():
                if to_next not in used:
                    heapq.heappush(edges, (cost2, to, to_next))
                    
    return mst

def solve():
    n = int(input())
    graph = {chr(i + 65): {} for i in range(n)}
    for i in range(n-1):
        data = input().split()
        star = data[0]
        m = int(data[1])
        for j in range(m):
            to_star = data[2+j*2]
            cost = int(data[3+j*2])
            graph[star][to_star], graph[to_star][star] = cost, cost
    mst = prim(graph, 'A')
    print(sum(x[2] for x in mst))
    
solve()
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

<img src="C:\Users\15744\AppData\Roaming\Typora\typora-user-images\image-20240508164641544.png" alt="image-20240508164641544" style="zoom:67%;" />



## 2. 学习总结和收获

==如果作业题目简单，有否额外练习题目，比如：OJ“2024spring每日选做”、CF、LeetCode、洛谷等网站题目。==

用栈来逆转！

dijkstra =(dp/greedy)= bfs + prior queue 

prim = greedy + tree

kruskal = greedy + disjoint set

### 图的dijkstra、prim、kruskal

```python
import heapq

class Graph:
    # ... 其他代码 ...

    def dijkstra(self, start_vertex):
        # 初始化距离和前驱顶点
        for vertex in self:
            vertex.distance = float('inf')
            vertex.previous = None

        start_vertex.distance = 0
        priority_queue = [(0, start_vertex)]

        while priority_queue:
            current_distance, current_vertex = heapq.heappop(priority_queue)

            if current_distance > current_vertex.distance:
                continue

            for neighbor in current_vertex.getNeighbor():
                weight = current_vertex.connectedTo[neighbor]
                distance = current_vertex.distance + weight

                if distance < neighbor.distance:
                    neighbor.distance = distance
                    neighbor.previous = current_vertex
                    heapq.heappush(priority_queue, (distance, neighbor))

    def prim(self, start_vertex):
        # 初始化顶点的距离和前驱顶点
        for vertex in self:
            vertex.distance = float('inf')
            vertex.previous = None

        start_vertex.distance = 0
        priority_queue = [(0, start_vertex)]
        visited = set()

        while priority_queue:
            current_distance, current_vertex = heapq.heappop(priority_queue)

            if current_vertex in visited:
                continue

            visited.add(current_vertex)

            for neighbor in current_vertex.getNeighbor():
                weight = current_vertex.connectedTo[neighbor]

                if neighbor not in visited and weight < neighbor.distance:
                    neighbor.distance = weight
                    neighbor.previous = current_vertex
                    heapq.heappush(priority_queue, (weight, neighbor))

    def kruskal(self):
        parent = {}
        rank = {}

        for vertex in self:
            parent[vertex] = vertex
            rank[vertex] = 0

        edges = []

        for vertex in self:
            for neighbor in vertex.getNeighbor():
                weight = vertex.connectedTo[neighbor]
                edges.append((weight, vertex, neighbor))

        edges.sort()

        minimum_spanning_tree = Graph()

        for edge in edges:
            weight, vertex1, vertex2 = edge

            if self.find(parent, vertex1) != self.find(parent, vertex2):
                minimum_spanning_tree.add_edge(vertex1.key, vertex2.key, weight)
                self.union(parent, rank, vertex1, vertex2)

        return minimum_spanning_tree

    def find(self, parent, vertex):
        if parent[vertex] != vertex:
            parent[vertex] = self.find(parent, parent[vertex])
        return parent[vertex]

    def union(self, parent, rank, vertex1, vertex2):
        root1 = self.find(parent, vertex1)
        root2 = self.find(parent, vertex2)

        if rank[root1] < rank[root2]:
            parent[root1] = root2
        elif rank[root1] > rank[root2]:
            parent[root2] = root1
        else:
            parent[root2] = root1
            rank[root1] += 1
```

这段代码基于前面的OOP算法，具体如下：

### 补：图的OOP

```python
class Vertex:
    def __init__(self, key):
        self.key = key
        self.connectedTo = {}
        self.color = "white"  # 顶点的颜色
        self.distance = float('inf')  # 顶点到起始顶点的距离，默认为无穷大
        self.previous = None  # 顶点在遍历中的前驱顶点
        self.disc = 0  # 顶点的发现时间
        self.fin = 0  # 顶点的完成时间

    def addNeighbor(self, nbr, weight=0):
        self.connectedTo[nbr] = weight

    def getNeighbor(self):
        return self.connectedTo.keys()


class Graph:
    def __init__(self):
        self.vertices = {}
        self.numVertices = 0
        self.numEdges = 0

    def add_vertex(self, key):
        self.numVertices += 1
        new_vertex = Vertex(key)
        self.vertices[key] = new_vertex
        return new_vertex

    def get_vertex(self, key):
        return self.vertices.get(key)

    def __len__(self):
        return self.numVertices

    def __contains__(self, key):
        return key in self.vertices

    def add_edge(self, f, t, weight=0):
        if f not in self.vertices:
            self.add_vertex(f)
        if t not in self.vertices:
            self.add_vertex(t)
        self.vertices[f].addNeighbor(self.vertices[t], weight)
        self.numEdges += 1

    def get_vertices(self):
        return self.vertices.keys()

    def __iter__(self):
        return iter(self.vertices.values())
