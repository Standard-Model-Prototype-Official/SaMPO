# Assignment #9: 图论：遍历，及 树算

Updated 1739 GMT+8 Apr 14, 2024

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

### 04081: 树的转换

http://cs101.openjudge.cn/dsapre/04081/



思路：

本来想用OOP，但是有计概方法就用了，是一种非常简单但是可以用来练习的dp。

代码

```python
#2300012610ljx
S = input()
h = a = b = 0
H = [0]*len(S)#dp
for s in S:
    if s == 'd':
        h += 1
        H[h] = H[h - 1] + 1
        a = max(a, h)
        b = max(b, H[h])
    else:
        h -= 1
        H[h] += 1
print('%d => %d' % (a, b))
```



代码运行截图 ==（至少包含有"Accepted"）==

<img src="C:\Users\15744\AppData\Roaming\Typora\typora-user-images\image-20240422105513391.png" alt="image-20240422105513391" style="zoom:67%;" />



### 08581: 扩展二叉树

http://cs101.openjudge.cn/dsapre/08581/



思路：

都是老知识点

代码

```python
#2300012610ljx
class BinaryTreeNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None
        
def build_tree(lst):
    if not lst:
        return None
    
    value = lst.pop()
    if value == '.':
        return None
    
    root = BinaryTreeNode(value)
    root.left = build_tree(lst)
    root.right = build_tree(lst)
    
    return root

def inorder(root):
    if not root:
        return []
    
    left = inorder(root.left)
    right = inorder(root.right)
    return left + [root.value] + right

def postorder(root):
    if not root:
        return []
    
    left = postorder(root.left)
    right = postorder(root.right)
    return left + right + [root.value]

lst = list(input())
root = build_tree(lst[::-1])
in_result = inorder(root)
post_result = postorder(root)
print(''.join(in_result))
print(''.join(post_result))
```



代码运行截图 ==（至少包含有"Accepted"）==

![image-20240422113109460](C:\Users\15744\AppData\Roaming\Typora\typora-user-images\image-20240422113109460.png)



### 22067: 快速堆猪

http://cs101.openjudge.cn/practice/22067/



思路：

维护了一个dp数组

代码

```python
#2300012610ljx
a = []
m = []

while True:
    try:
        s = input().split()
        
        if s[0] == 'pop':
            if a:
                a.pop()
                if m:
                    m.pop()
        elif s[0] == 'min':
            if m:
                print(m[-1])
        else:
            h = int(s[1])
            a.append(h)
            if not m:
                m.append(h)
            else:
                k = m[-1]
                m.append(min(k, h))
    except EOFError:
        break
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

<img src="C:\Users\15744\AppData\Roaming\Typora\typora-user-images\image-20240422114742002.png" alt="image-20240422114742002" style="zoom:67%;" />



### 04123: 马走日

dfs, http://cs101.openjudge.cn/practice/04123



思路：

dfs

代码

```python
#2300012610ljx
ans = 0
def dfs(mx, x, y, dirs, step):
    global ans
    n = len(mx)
    m = len(mx[0])
    
    if step == n*m:
        ans += 1
        return
    
    mx[x][y] = 1
    for dx, dy in dirs:
        nx, ny = x+dx, y+dy
        if 0<=nx<n and 0<=ny<m and board[nx][ny] == 0:
            dfs(mx, nx, ny, dirs, step+1)
            mx[nx][ny] = 0
            
T = int(input())
dirs = [(1, -2), (2, -1), (2, 1), \
        (1, 2), (-1, 2), (-2, 1), \
        (-2, -1), (-1, -2)]
for _ in range(T):
    ans = 0
    n, m, x, y = map(int, input().split())
    board = [[0]*m for _ in range(n)]
    board[x][y] = 1
    dfs(board, x, y, dirs, 1)
    print(ans)
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

<img src="C:\Users\15744\AppData\Roaming\Typora\typora-user-images\image-20240423153914430.png" alt="image-20240423153914430" style="zoom:67%;" />



### 28046: 词梯

bfs, http://cs101.openjudge.cn/practice/28046/



思路：

图的bfs算法

代码

```python
#2300012610ljx
import sys
from collections import deque

class Graph:
    def __init__(self):
        self.vertices = {}
        self.num_vertices = 0
        
    def add_vertex(self, key):
        self.num_vertices += 1
        new_vertex = Vertex(key)
        self.vertices[key] = new_vertex
        return new_vertex
    
    def get_vertex(self, n):
        if n in self.vertices:
            return self.vertices[n]
        else:
            return None
        
    def __len__(self):
        return self.num_vertices
    
    def __contains__(self, n):
        return n in self.vertices
    
    def add_edge(self, f, t, cost=0):
        if f not in self.vertices:
            nv = self.add_vertex(f)
        if t not in self.vertices:
            nv = self.add_vertex(t)
        self.vertices[f].add_neighbor(self.vertices[t], cost)
        
    def get_vertices(self):
        return list(self.vertices.keys())
    
    def __iter__(self):
        return iter(self.vertices.values())
    
    
class Vertex:
    def __init__(self, num):
        self.key = num
        self.connectedTo = {}
        self.color = 'white'
        self.distance = sys.maxsize
        self.previous = None
        self.disc = 0
        self.fin = 0
        
    def add_neighbor(self, nbr, weight=0):
        self.connectedTo[nbr] = weight
        
    def get_neighbors(self):
        return self.connectedTo.keys()
    
def build_graph(all_words):
    buckets = {}
    the_graph = Graph()
    
    for line in all_words:
        word = line.strip()
        for i, _ in enumerate(word):
            bucket = f"{word[:i]}_{word[i+1:]}"
            buckets.setdefault(bucket, set()).add(word)
            
    for similar_words in buckets.values():
        for word1 in similar_words:
            for word2 in similar_words - {word1}:
                the_graph.add_edge(word1, word2)

    return the_graph

def bfs(start, end):
    start.distnce = 0
    start.previous = None
    vert_queue = deque()
    vert_queue.append(start)
    while len(vert_queue) > 0:
        current = vert_queue.popleft()
        
        if current == end:
            return True
        
        for neighbor in current.get_neighbors():
            if neighbor.color == "white":
                neighbor.color = "gray"
                neighbor.distance = current.distance + 1
                neighbor.previous = current
                vert_queue.append(neighbor)
        current.color = "black"
        
    return False

def traverse(starting_vertex):
    ans = []
    current = starting_vertex
    while (current.previous):
        ans.append(current.key)
        current = current.previous
    ans.append(current.key)
    
    return ans

n = int(input())
all_words = []
for _ in range(n):
    all_words.append(input().strip())
    
g = build_graph(all_words)

s, e = input().split()
start, end = g.get_vertex(s), g.get_vertex(e)
if start is None or end is None:
    print('NO')
    exit(0)
    
if bfs(start, end):
    ans = traverse(end)
    print(*ans[::-1])
else:
    print('NO')
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

<img src="C:\Users\15744\AppData\Roaming\Typora\typora-user-images\image-20240423231729343.png" alt="image-20240423231729343" style="zoom:67%;" />



### 28050: 骑士周游

dfs, http://cs101.openjudge.cn/practice/28050/



思路：



代码

```python
#2300012610ljx
import sys

class Graph:
    def __init__(self):
        self.vertices = {}
        self.num_vertices = 0
        
    def add_vertex(self, key):
        self.num_vertices += 1
        new_vertex = Vertex(key)
        self.vertices[key] = new_vertex
        return new_vertex
    
    def get_vertex(self, n):
        if n in self.vertices:
            return self.vertices[n]
        else:
            return None
        
    def __len__(self):
        return self.num_vertices
    
    def __contains__(self, n):
        return n in self.vertices
    
    def add_edge(self, f, t, cost=0):
        if f not in self.vertices:
            nv = self.add_vertex(f)
        if t not in self.vertices:
            nv = self.add_vertex(t)
        self.vertices[f].add_neighbor(self.vertices[t], cost)
        
    def get_vertices(self):
        return list(self.vertices.keys())
    
    def __iter__(self):
        return iter(self.vertices.values())
    
    
class Vertex:
    def __init__(self, num):
        self.key = num
        self.connectedTo = {}
        self.color = 'white'
        self.distance = sys.maxsize
        self.previous = None
        self.disc = 0
        self.fin = 0
        
    def __lt__(self, o):
        return self.key < o.key
        
    def add_neighbor(self, nbr, weight=0):
        self.connectedTo[nbr] = weight
        
    def get_neighbors(self):
        return self.connectedTo.keys()
        
    def __str__(self):
        return str(self.key) + ":color" + self.color + ":disc" + str(self.disc) + ":fin" + str(
            self.fin) + ":dist" + str(self.distance) + ":pred \n\t[" + str(self.previous) + "]\n"

def knight_graph(board_size):
    kt_graph = Graph()
    for row in range(board_size):
        for col in range(board_size):
            node_id = pos_to_node_id(row, col, board_size)
            new_positions = gen_legal_moves(row, col, board_size)
            for row2, col2 in new_positions:
                other_node_id = pos_to_node_id(row2, col2, board_size)
                kt_graph.add_edge(node_id, other_node_id)
    return kt_graph

def pos_to_node_id(x, y, bdsize):
    return x * bdsize + y

def gen_legal_moves(row, col, board_size):
    new_moves = []
    move_offsets = [
        (-1, -2),
        (-1, 2),
        (-2, -1),
        (-2, 1),
        (1, -2),
        (1, 2),
        (2, -1),
        (2, 1),
    ]
    for r_off, c_off in move_offsets:
        if (
            0 <= row + r_off < board_size
            and 0 <= col + c_off < board_size
        ):
            new_moves.append((row + r_off, col + c_off))
    return new_moves

def knight_tour(n, path, u, limit):
    u.color = "gray"
    path.append(u)
    if n < limit:
        neighbors = ordered_by_avail(u)
        i = 0
        
        for nbr in neighbors:
            if nbr.color == "white" and \
                knight_tour(n + 1, path, nbr, limit):
                return True
        else:
            path.pop()
            u.color = "white"
            return False
    else:
        return True
    
def ordered_by_avail(n):
    res_list = []
    for v in n.get_neighbors():
        if v.color == "white":
            c = 0
            for w in v.get_neighbors():
                if w.color == "white":
                    c += 1
            res_list.append((c, v))
    res_list.sort(key = lambda x: x[0])
    return [y[1] for y in res_list]

def main():
    bdsize = int(input())
    *start_pos, = map(int, input().split())  # 起始位置
    g = knight_graph(bdsize)
    start_vertex = g.get_vertex(pos_to_node_id(start_pos[0], start_pos[1], bdsize))
    if start_vertex is None:
        print("fail")
        exit(0)

    tour_path = []
    done = knight_tour(0, tour_path, start_vertex, bdsize * bdsize-1)
    if done:
        print("success")
    else:
        print("fail")

    exit(0)

    cnt = 0
    for vertex in tour_path:
        cnt += 1
        if cnt % bdsize == 0:
            print()
        else:
            print(vertex.key, end=" ")

if __name__ == '__main__':
    main()
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

<img src="C:\Users\15744\AppData\Roaming\Typora\typora-user-images\image-20240423234952355.png" alt="image-20240423234952355" style="zoom:67%;" />



## 2. 学习总结和收获

==如果作业题目简单，有否额外练习题目，比如：OJ“2024spring每日选做”、CF、LeetCode、洛谷等网站题目。==

有些题目乍一看是数算题目，但是脑子转得快的话完全可以用计概办法做出来，毕竟检验正确与否只看答案。

图的dfs、bfs和树的dfs、bfs如出一辙，只是有环的缘故，需要以某种方式打上标记。

整理了比较一般的图的oop写法如下

### 图的OOP写法及dfs、bfs

```python
from collections import deque

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

    def dfs(self, start_vertex):
        start_vertex.color = "gray"
        for neighbor in start_vertex.getNeighbor():
            if neighbor.color == "white":
                self.dfs(neighbor)
        start_vertex.color = "black"

    def bfs(self, start_vertex):
        queue = deque()
        start_vertex.color = "gray"
        queue.append(start_vertex)

        while queue:
            current_vertex = queue.popleft()
            for neighbor in current_vertex.getNeighbor():
                if neighbor.color == "white":
                    neighbor.color = "gray"
                    queue.append(neighbor)
            current_vertex.color = "black"
```



