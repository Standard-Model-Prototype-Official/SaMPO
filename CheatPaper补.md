### Floyd算法

```python
import math

class Graph:
    def __init__(self, n):
        self.n = n
        self.adj_matrix = [[math.inf for _ in range(n)] for _ in range(n)]
        self.prev = [[None for _ in range(n)] for _ in range(n)]
        
        # 初始化对角线元素为0
        for i in range(n):
            self.adj_matrix[i][i] = 0
    
    def add_edge(self, u, v, weight):
        self.adj_matrix[u][v] = weight
        self.prev[u][v] = u
    
    def floyd(self):
        for k in range(self.n):
            for i in range(self.n):
                for j in range(self.n):
                    if self.adj_matrix[i][k] + self.adj_matrix[k][j] < self.adj_matrix[i][j]:
                        self.adj_matrix[i][j] = self.adj_matrix[i][k] + self.adj_matrix[k][j]
                        self.prev[i][j] = self.prev[k][j]
    
    def print_shortest_path(self, s, t):
        if self.adj_matrix[s][t] == math.inf:
            print("No path from vertex {0} to vertex {1}".format(s, t))
        else:
            path = [t]
            while path[0] != s:
                path.insert(0, self.prev[s][path[0]])
            print("Shortest path from vertex {0} to vertex {1} is: {2}".format(s, t, " -> ".join(map(str, path))))

# 测试用例
g = Graph(4)
g.add_edge(0, 1, 5)
g.add_edge(0, 3, 10)
g.add_edge(1, 2, 3)
g.add_edge(2, 3, 1)
g.add_edge(2, 0, 2)
g.add_edge(3, 1, 2)

g.floyd()
g.print_shortest_path(0, 3)
g.print_shortest_path(2, 1)
```

#### Dijkstra算法

```python
from collections import defaultdict
import heapq

def dijkstra(graph, start, end):
    """
    使用 Dijkstra 算法查找从 start 节点到 end 节点的最短路径和距离。

    参数:
    graph (dict): 图的邻接表表示,格式为 {node: {neighbor: weight, ...}, ...}
    start (str): 起始节点
    end (str): 目标节点

    返回:
    path (list): 从 start 到 end 的最短路径
    distance (int): 从 start 到 end 的最短距离
    """
    # 初始化
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    heap = [(0, start)]
    prev_nodes = {start: None}

    while heap:
        current_distance, current_node = heapq.heappop(heap)

        # 如果当前节点是目标节点,则返回最短路径和距离
        if current_node == end:
            path = []
            while current_node is not None:
                path.append(current_node)
                current_node = prev_nodes[current_node]
            path.reverse()
            return path, current_distance

        # 如果当前节点的距离已经大于已知的最短距离,则跳过
        if current_distance > distances[current_node]:
            continue

        # 更新当前节点的邻居节点的距离
        for neighbor, weight in graph[current_node].items():
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                prev_nodes[neighbor] = current_node
                heapq.heappush(heap, (distance, neighbor))

    # 如果没有找到目标节点,返回 None
    return None, None

# 示例使用
graph = {
    'A': {'B': 5, 'C': 1},
    'B': {'A': 5, 'C': 2, 'D': 1},
    'C': {'A': 1, 'B': 2, 'D': 4, 'E': 8},
    'D': {'B': 1, 'C': 4, 'E': 3, 'F': 6},
    'E': {'C': 8, 'D': 3},
    'F': {'D': 6}
}

path, distance = dijkstra(graph, 'A', 'F')
print(f"最短路径: {' -> '.join(path)}")
print(f"最短距离: {distance}")
```

#### prim

```python
from collections import defaultdict
import heapq

def prim(graph, start):
    """
    使用 Prim 算法找到最小生成树。

    参数:
    graph (dict): 图的邻接表表示,格式为 {node: {neighbor: weight, ...}, ...}
    start (str): 起始节点

    返回:
    mst (list): 最小生成树的边列表
    """
    # 初始化
    mst = []
    visited = set()
    heap = [(0, start, None)]  # (权重, 节点, 父节点)

    while heap:
        weight, node, parent = heapq.heappop(heap)

        # 如果节点未访问过,则添加到最小生成树中
        if node not in visited:
            visited.add(node)
            if parent is not None:
                mst.append((parent, node, weight))

            # 将该节点的邻居加入堆中
            for neighbor, w in graph[node].items():
                heapq.heappush(heap, (w, neighbor, node))

    return mst

# 示例使用
graph = {
    'A': {'B': 5, 'C': 1},
    'B': {'A': 5, 'C': 2, 'D': 1},
    'C': {'A': 1, 'B': 2, 'D': 4, 'E': 8},
    'D': {'B': 1, 'C': 4, 'E': 3, 'F': 6},
    'E': {'C': 8, 'D': 3},
    'F': {'D': 6}
}

mst = prim(graph, 'A')
print("最小生成树的边:")
for edge in mst:
    print(f"{edge[0]} - {edge[1]}: {edge[2]}")
```

#### krustal

```python
from collections import defaultdict
import heapq

def prim(graph, start):
    """
    使用 Prim 算法找到最小生成树。

    参数:
    graph (dict): 图的邻接表表示,格式为 {node: {neighbor: weight, ...}, ...}
    start (str): 起始节点

    返回:
    mst (list): 最小生成树的边列表
    """
    # 初始化
    mst = []
    visited = set()
    heap = [(0, start, None)]  # (权重, 节点, 父节点)

    while heap:
        weight, node, parent = heapq.heappop(heap)

        # 如果节点未访问过,则添加到最小生成树中
        if node not in visited:
            visited.add(node)
            if parent is not None:
                mst.append((parent, node, weight))

            # 将该节点的邻居加入堆中
            for neighbor, w in graph[node].items():
                heapq.heappush(heap, (w, neighbor, node))

    return mst

# 示例使用
graph = {
    'A': {'B': 5, 'C': 1},
    'B': {'A': 5, 'C': 2, 'D': 1},
    'C': {'A': 1, 'B': 2, 'D': 4, 'E': 8},
    'D': {'B': 1, 'C': 4, 'E': 3, 'F': 6},
    'E': {'C': 8, 'D': 3},
    'F': {'D': 6}
}

mst = prim(graph, 'A')
print("最小生成树的边:")
for edge in mst:
    print(f"{edge[0]} - {edge[1]}: {edge[2]}")
```

#### 判断有无环

```python
from collections import defaultdict

def has_cycle(graph):
    """
    使用 DFS 遍历判断图是否有环。

    参数:
    graph (dict): 图的邻接表表示,格式为 {node: [neighbor, ...], ...}

    返回:
    bool: 如果图有环,返回 True,否则返回 False
    """
    visited = set()
    visiting = set()

    def dfs(node):
        visiting.add(node)
        for neighbor in graph[node]:
            if neighbor in visiting or (neighbor in visited and neighbor in graph[node]):
                return True
            if neighbor not in visited:
                if dfs(neighbor):
                    return True
        visiting.remove(node)
        visited.add(node)
        return False

    for node in graph:
        if node not in visited:
            if dfs(node):
                return True

    return False

# 示例使用
graph = {
    'A': ['B', 'C'],
    'B': ['C', 'D'],
    'C': ['D'],
    'D': ['C', 'E'],
    'E': [],
}

print(has_cycle(graph))  # 输出 True

graph = {
    'A': ['B'],
    'B': ['C'],
    'C': ['D'],
    'D': [],
}

print(has_cycle(graph))  # 输出 False
```

#### 动态中位数

```python
#动态中位数
import heapq
def main():
    lst=list(map(int,input().split()))
    n=len(lst)
    ans=[]
    bigheap=[]
    smallheap=[]
    heapq.heapify(bigheap)
    heapq.heapify(smallheap)
    for i in range(n):
        if not smallheap or -smallheap[0]>=lst[i]:
            heapq.heappush(smallheap,-lst[i])
        else:
            heapq.heappush(bigheap,lst[i])
        if len(bigheap)>len(smallheap):
            heapq.heappush(smallheap,-heapq.heappop(bigheap))
        if len(smallheap)>len(bigheap)+1:
            heapq.heappush(bigheap,-heapq.heappop(smallheap))
        if i%2==0:
            ans.append(-smallheap[0])
    print(len(ans))
    print(' '.join(map(str,ans)))
t=int(input())
for i in range(t):
    main()
```

#### 调度场

```python
class Token:
    def __init__(self, value, type):
        self.value = value
        self.type = type

def shunting_yard(expression):
    """
    将中缀表达式转换为后缀表达式(逆波兰表达式)。

    参数:
    expression (str): 中缀表达式

    返回:
    list: 后缀表达式
    """
    operators = {'+': 1, '-': 1, '*': 2, '/': 2, '^': 3}
    output = []
    stack = []

    for token in expression.split():
        if token.isdigit():
            output.append(Token(int(token), 'operand'))
        elif token in operators:
            while stack and stack[-1].type == 'operator' and operators[stack[-1].value] >= operators[token]:
                output.append(stack.pop())
            stack.append(Token(token, 'operator'))
        elif token == '(':
            stack.append(Token(token, 'left_paren'))
        elif token == ')':
            while stack and stack[-1].value != '(':
                output.append(stack.pop())
            if stack and stack[-1].value == '(':
                stack.pop()

    while stack:
        output.append(stack.pop())

    return output

# 示例用法
infix_expression = "2 + 3 * 4 - 5 / 6"
postfix_expression = shunting_yard(infix_expression)

print("中缀表达式:", infix_expression)
print("后缀表达式:", [token.value for token in postfix_expression])
```

#### 二分查找

```python
# hi:不可行最小值， lo:可行最大值
lo, hi, ans = 0, max(lst), 0
while lo + 1 < hi:
    mid = (lo + hi) // 2
    # print(lo, hi, mid)
    if check(mid): # 返回True，是因为num>m，是确定不合适
        ans = mid
        lo = mid # 所以lo可以置为 mid + 1。
    else:
        hi = mid
#print(lo)
print(ans)
```

#### kosaraju's

```python
from collections import defaultdict

class Graph:
    def __init__(self, vertices):
        self.graph = defaultdict(list)
        self.V = vertices

    def addEdge(self, u, v):
        self.graph[u].append(v)

    def _dfs(self, v, visited, stack):
        visited[v] = True
        for neighbour in self.graph[v]:
            if not visited[neighbour]:
                self._dfs(neighbour, visited, stack)
        stack.append(v)

    def _transpose(self):
        g = Graph(self.V)
        for i in self.graph:
            for j in self.graph[i]:
                g.addEdge(j, i)
        return g

    def _fillOrder(self, v, visited, stack):
        visited[v] = True
        for neighbour in self.graph[v]:
            if not visited[neighbour]:
                self._fillOrder(neighbour, visited, stack)
        stack.append(v)

    def _dfsUtil(self, v, visited):
        visited[v] = True
        print(v, end=' ')
        for neighbour in self.graph[v]:
            if not visited[neighbour]:
                self._dfsUtil(neighbour, visited)

    def printSCCs(self):
        stack = []
        visited = [False] * self.V

        for i in range(self.V):
            if not visited[i]:
                self._fillOrder(i, visited, stack)

        gr = self._transpose()

        visited = [False] * self.V

        while stack:
            i = stack.pop()
            if not visited[i]:
                gr._dfsUtil(i, visited)
                print("")

# 示例使用
g = Graph(5)
g.addEdge(1, 0)
g.addEdge(0, 2)
g.addEdge(2, 1)
g.addEdge(0, 3)
g.addEdge(3, 4)

print("Strongly Connected Components:")
g.printSCCs()
```

