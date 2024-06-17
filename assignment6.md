# Assignment #6: "树"算：Huffman,BinHeap,BST,AVL,DisjointSet

Updated 2214 GMT+8 March 24, 2024

2024 spring, Complied by ==罗景轩，地空==



**说明：**

1）这次作业内容不简单，耗时长的话直接参考题解。

2）请把每个题目解题思路（可选），源码Python, 或者C++（已经在Codeforces/Openjudge上AC），截图（包含Accepted），填写到下面作业模版中（推荐使用 typora https://typoraio.cn ，或者用word）。AC 或者没有AC，都请标上每个题目大致花费时间。

3）提交时候先提交pdf文件，再把md或者doc文件上传到右侧“作业评论”。Canvas需要有同学清晰头像、提交文件有pdf、"作业评论"区有上传的md或者doc附件。

4）如果不能在截止前提交作业，请写明原因。



**编程环境**

操作系统：Windows 11 家庭中文版 22H2 22621.2283 

Python编程环境：Spyder IDE 5.2.2, PyCharm 2023.1.4 (Professional Edition)



## 1. 题目

### 22275: 二叉搜索树的遍历

http://cs101.openjudge.cn/practice/22275/



思路：

第一个就是根节点，然后序列之后第一个比根大的树就是右子树的根节点，第二个就是左子树的根节点；或者考虑顺序排列就是中序遍历，然后复用上次作业的代码。

代码

```python
#2300012610ljx
class Node():
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None
        
def buildtree(preorder):
    if len(preorder) == 0:
        return None
    
    node = Node(preorder[0])
    
    idx = len (preorder)
    for i in range(1, len(preorder)):
        if preorder[i] > preorder[0]:
            idx = i
            break
    node.left = buildtree(preorder[1:idx])
    node.right = buildtree(preorder[idx:])
    
    return node

def postorder(node):
    if not node:
        return []
    result = []
    result += postorder(node.left)
    result += postorder(node.right)
    result += node.value,
    
    return result

n = int(input())
preorder = list(map(int, input().split()))
print(*postorder(buildtree(preorder)))

```



代码运行截图 ==（至少包含有"Accepted"）==

<img src="C:\Users\15744\AppData\Roaming\Typora\typora-user-images\image-20240330175215746.png" alt="image-20240330175215746" style="zoom:67%;" />



### 05455: 二叉搜索树的层次遍历

http://cs101.openjudge.cn/practice/05455/



思路：

先建树，然后按上次作业的bfs输出

代码

```python
#2300012610ljx
class Node():
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None
        
def insert(node, root):
    if not root:
        return node
    
    if node.value < root.value:
        root.left = insert(node, root.left)
    elif node.value > root.value:
        root.right = insert(node, root.right)
    return root
        
def levelorder(root):
    source = [root]
    answer = []
    while source:
        node = source.pop(0)
        answer += node.value,
        if node.left:
            source += node.left,
        if node.right:
            source += node.right,
    return answer

array = list(map(int, input().split()))
root = Node(array.pop(0))
while array:
    node = Node(array.pop(0))
    root = insert(node, root)
print(*levelorder(root))

```



代码运行截图 ==（至少包含有"Accepted"）==

<img src="C:\Users\15744\AppData\Roaming\Typora\typora-user-images\image-20240330180909031.png" alt="image-20240330180909031" style="zoom:67%;" />



### 04078: 实现堆结构

http://cs101.openjudge.cn/practice/04078/

练习自己写个BinHeap。当然机考时候，如果遇到这样题目，直接import heapq。手搓栈、队列、堆、AVL等，考试前需要搓个遍。



思路：

堆，虚拟的树。

代码

```python
#2300012610ljx
class BinHeap:
    def __init__(self):
        self.fact = [0]
        self.size = 0
    
    def Benjamin(self, i):
        if (j := i * 2) + 1 > self.size:
            return j
        else:
            return j if self.fact[j] < self.fact[j + 1] else j + 1
    
    def percUp(self, i):
        while (j := i // 2) > 0:
            if self.fact[i] < self.fact[j]:
                self.fact[i], self.fact[j] = self.fact[j], self.fact[i]
            i = j
            
    def percDown(self, i):
        while i * 2 <= self.size:
            j = self.Benjamin(i)
            if self.fact[i] > self.fact[j]:
                self.fact[i], self.fact[j] = self.fact[j], self.fact[i]
            i = j
            
    def insert(self, k):
        self.fact += k,
        self.size += 1
        self.percUp(self.size)
            
    def pop(self):
        result = self.fact[1]
        self.fact[1] = self.fact[self.size]
        self.size -= 1
        self.fact.pop()
        self.percDown(1)
        return result
    
    def buildHeap(self, array):
        self.size = len(array)
        i = self.size // 2
        self.fact = [0] + array[:]
        while i > 0:
            self.percDown(i)
            i -= 1
            
n = int(input())
heap = BinHeap()
for _ in range(n):
    line = input()
    if line[0] == '1':
        heap.insert(int(line.split()[1]))
    else:
        print(heap.pop())
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

<img src="C:\Users\15744\AppData\Roaming\Typora\typora-user-images\image-20240401080421425.png" alt="image-20240401080421425" style="zoom:67%;" />



### 22161: 哈夫曼编码树

http://cs101.openjudge.cn/practice/22161/



思路：

哈夫曼编码树

代码

```python
import heapq

class Node:
    def __init__(self, weight, char=None):
        self.weight = weight
        self.char = char
        self.left = None
        self.right = None
        
    def __lt__(self, other):
        if self.weight == other.weight:
            return self.char < other.char
        return self.weight < other.weight
    
def build_huffman_tree(characters):
    heap = []
    for char, weight in characters.items():
        heapq.heappush(heap, Node(weight, char))
        
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = Node(left.weight + right.weight, min(left.char, right.char))
        merged.left = left
        merged.right = right
        heapq.heappush(heap, merged)
        
    return heap[0]

def encode_huffman_tree(root):
    codes = {}
    
    def traverse(node, code):
        if not (node.left or node.right):
            codes[node.char] = code
        else:
            traverse(node.left, code + '0')
            traverse(node.right, code + '1')
            
    traverse(root, '')
    return codes

def huffman_encoding(codes, string):
    encoded = ''
    for char in string:
        encoded += codes[char]
    return encoded

def huffman_decoding(root, encoded_string):
    decoded = ''
    node = root
    for bit in encoded_string:
        if bit == '0':
            node = node.left
        else:
            node = node.right
            
        if node.left is None and node.right is None:
            decoded += node.char
            node = root
    return decoded

n = int(input())
characters = {}
for _ in range(n):
    char, weight = input().split()
    characters[char] = int(weight)
    
huffman_tree =build_huffman_tree(characters)

codes = encode_huffman_tree(huffman_tree)

while True:
    try:
        line = input()
        if line[0] in ('0', '1'):
            print(huffman_decoding(huffman_tree, line))
        else:
            print(huffman_encoding(codes, line))
    except:
        break
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

<img src="C:\Users\15744\AppData\Roaming\Typora\typora-user-images\image-20240401204257351.png" alt="image-20240401204257351" style="zoom:67%;" />



### 晴问9.5: 平衡二叉树的建立

https://sunnywhy.com/sfbj/9/5/359



思路：

AVL树

代码

```python
class Node:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None
        self.height = 1
        
class AVL:
    def __init__(self):
        self.root = None
        
    def insert(self, value):
        if not self.root:
            self.root = Node(value)
        else:
            self.root = self._insert(value, self.root)
            
    def _insert(self, value, node):
        if not node:
            return Node(value)
        elif value < node.value:
            node.left = self._insert(value, node.left)
        else:
            node.right = self._insert(value, node.right)
            
        node.height = 1 + max(self._get_height(node.left), self._get_height(node.right))
        
        balance = self._get_balance(node)
        
        if balance > 1:
            if value < node.left.value:
                return self._rotate_right(node)
            else:
                node.left = self._rotate_left(node.left)
                return self._rotate_right(node)
            
        if balance < -1:
            if value > node.right.value:
                return self._rotate_left(node)
            else:
                node.right = self._rotate_right(node.right)
                return self._rotate_left(node)
            
        return node
    
    def _get_height(self, node):
        if not node:
            return 0
        return node.height
    
    def _get_balance(self, node):
        if not node:
            return 0
        return self._get_height(node.left) - self._get_height(node.right)
    
    def _rotate_left(self, z):
        y = z.right
        T2 = y.left
        y.left = z
        z.right = T2
        z.height = 1 + max(self._get_height(z.left), self._get_height(z.right))
        y.height = 1 + max(self._get_height(y.left), self._get_height(y.right))
        return y
    
    def _rotate_right(self, y):
        x = y.left
        T2 = x.right
        x.right = y
        y.left = T2
        y.height = 1 + max(self._get_height(y.left), self._get_height(y.right))
        x.height = 1 + max(self._get_height(x.left), self._get_height(x.right))
        return x
    
    def preorder(self):
        return self._preorder(self.root)
    
    def _preorder(self, node):
        if not node:
            return []
        return [node.value] + self._preorder(node.left) + self._preorder(node.right)
    
n = int(input())
array = list(map(int, input().split()))

avl = AVL()
for value in array:
    avl.insert(value)
    
print(*avl.preorder())

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

<img src="C:\Users\15744\AppData\Roaming\Typora\typora-user-images\image-20240402172021194.png" alt="image-20240402172021194" style="zoom:67%;" />



### 02524: 宗教信仰

http://cs101.openjudge.cn/practice/02524/



思路：

并查集

代码

```python
def init_set(n):
    return list(range(n))

def get_father(x, father):
    if father[x] != x:
        father [x] = get_father(father[x], father)
    return father[x]

def join(x, y, father):
    fx = get_father(x, father)
    fy = get_father(y, father)
    if fx == fy:
        return
    father[fx] = fy
    
def is_same(x, y, father):
    return get_father(x, father) == get_father(y, father)

case_num = 0
while True:
    n, m = map(int, input().split())
    if n == 0 and m == 0:
        break
    count = 0
    father = init_set(n)
    for _ in range(m):
        s1, s2 = map(int, input().split())
        join(s1-1, s2-1, father)
    for i in range(n):
        if father[i] == i:
            count += 1
    case_num += 1
    print(f'Case {case_num}: {count}')
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![image-20240402193229455](C:\Users\15744\AppData\Roaming\Typora\typora-user-images\image-20240402193229455.png)



## 2. 学习总结和收获

==如果作业题目简单，有否额外练习题目，比如：OJ“2024spring每日选做”、CF、LeetCode、洛谷等网站题目。==

树的进阶内容，内含堆、哈夫曼编码树、AVL、并查集

AVL特别长。非常好平衡因子，使我的子树旋转。

RR和LL直接转就好。但是RL和LR呢：

你的母节点比较不平衡，但是呢，你的子节点又弥补了这一部分，如果做旋转，可能会显得你还是比较不平衡，可能会有一些RL、LR互变的情况。现在最好的办法就是：在做旋转的同时，给你的子树提前做一个反向旋转的操作。

### 堆

```python
class BinHeap:
    def __init__(self):
        self.fact = [0]
        self.size = 0
    
    def Benjamin(self, i):
        if (j := i * 2) + 1 > self.size:
            return j
        else:
            return j if self.fact[j] < self.fact[j + 1] else j + 1
    
    def percUp(self, i):
        while (j := i // 2) > 0:
            if self.fact[i] < self.fact[j]:
                self.fact[i], self.fact[j] = self.fact[j], self.fact[i]
            i = j
            
    def percDown(self, i):
        while i * 2 <= self.size:
            j = self.Benjamin(i)
            if self.fact[i] > self.fact[j]:
                self.fact[i], self.fact[j] = self.fact[j], self.fact[i]
            i = j
            
    def insert(self, k):
        self.fact += k,
        self.size += 1
        self.percUp(self.size)
            
    def pop(self):
        result = self.fact[1]
        self.fact[1] = self.fact[self.size]
        self.size -= 1
        self.fact.pop()
        self.percDown(1)
        return result
    
    def buildHeap(self, array):
        self.size = len(array)
        i = self.size // 2
        self.fact = [0] + array[:]
        while i > 0:
            self.percDown(i)
            i -= 1
```

### huffman树

```python
import heapq

class Node:
    def __init__(self, weight, char=None):
        self.weight = weight
        self.char = char
        self.left = None
        self.right = None
        
    def __lt__(self, other):
        if self.weight == other.weight:
            return self.char < other.char
        return self.weight < other.weight
    
def build_huffman_tree(characters):
    heap = []
    for char, weight in characters.items():
        heapq.heappush(heap, Node(weight, char))
        
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = Node(left.weight + right.weight, min(left.char, right.char))
        merged.left = left
        merged.right = right
        heapq.heappush(heap, merged)
        
    return heap[0]

def encode_huffman_tree(root):
    codes = {}
    
    def traverse(node, code):
        if not (node.left or node.right):
            codes[node.char] = code
        else:
            traverse(node.left, code + '0')
            traverse(node.right, code + '1')
            
    traverse(root, '')
    return codes

def huffman_encoding(codes, string):
    encoded = ''
    for char in string:
        encoded += codes[char]
    return encoded

def huffman_decoding(root, encoded_string):
    decoded = ''
    node = root
    for bit in encoded_string:
        if bit == '0':
            node = node.left
        else:
            node = node.right
            
        if node.left is None and node.right is None:
            decoded += node.char
            node = root
    return decoded
```

### AVL平衡二叉树

```python
class Node:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None
        self.height = 1
        
class AVL:
    def __init__(self):
        self.root = None
        
    def insert(self, value):
        if not self.root:
            self.root = Node(value)
        else:
            self.root = self._insert(value, self.root)
            
    def _insert(self, value, node):
        if not node:
            return Node(value)
        elif value < node.value:
            node.left = self._insert(value, node.left)
        else:
            node.right = self._insert(value, node.right)
            
        node.height = 1 + max(self._get_height(node.left), self._get_height(node.right))
        
        balance = self._get_balance(node)
        
        if balance > 1:
            if value < node.left.value:
                return self._rotate_right(node)
            else:
                node.left = self._rotate_left(node.left)
                return self._rotate_right(node)
            
        if balance < -1:
            if value > node.right.value:
                return self._rotate_left(node)
            else:
                node.right = self._rotate_right(node.right)
                return self._rotate_left(node)
            
        return node
    
    def _get_height(self, node):
        if not node:
            return 0
        return node.height
    
    def _get_balance(self, node):
        if not node:
            return 0
        return self._get_height(node.left) - self._get_height(node.right)
    
    def _rotate_left(self, z):
        y = z.right
        T2 = y.left
        y.left = z
        z.right = T2
        z.height = 1 + max(self._get_height(z.left), self._get_height(z.right))
        y.height = 1 + max(self._get_height(y.left), self._get_height(y.right))
        return y
    
    def _rotate_right(self, y):
        x = y.left
        T2 = x.right
        x.right = y
        y.left = T2
        y.height = 1 + max(self._get_height(y.left), self._get_height(y.right))
        x.height = 1 + max(self._get_height(x.left), self._get_height(x.right))
        return x
    
    def preorder(self):
        return self._preorder(self.root)
    
    def _preorder(self, node):
        if not node:
            return []
        return [node.value] + self._preorder(node.left) + self._preorder(node.right)
```

### 并查集

```python
class DisjSet:
    def __init__(self, n):
        f.rank = [1] * n
        self.parent = [i for i in range(n)]

    def find(self, x):
        if (self.parent[x] != x):
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def Union(self, x, y):
        xset = self.find(x)
        yset = self.find(y)
        if xset == yset:
            return

        if self.rank[xset] < self.rank[yset]:
            self.parent[xset] = yset
        elif self.rank[xset] > self.rank[yset]:
            self.parent[yset] = xset
        else:
            self.parent[yset] = xset
            self.rank[xset] = self.rank[xset] + 1
```

