# Assignment #5: "树"算：概念、表示、解析、遍历

Updated 2124 GMT+8 March 17, 2024

2024 spring, Complied by ==罗景轩，地空==



**说明：**

1）The complete process to learn DSA from scratch can be broken into 4 parts:

Learn about Time complexities, learn the basics of individual Data Structures, learn the basics of Algorithms, and practice Problems.

2）请把每个题目解题思路（可选），源码Python, 或者C++（已经在Codeforces/Openjudge上AC），截图（包含Accepted），填写到下面作业模版中（推荐使用 typora https://typoraio.cn ，或者用word）。AC 或者没有AC，都请标上每个题目大致花费时间。

3）提交时候先提交pdf文件，再把md或者doc文件上传到右侧“作业评论”。Canvas需要有同学清晰头像、提交文件有pdf、"作业评论"区有上传的md或者doc附件。

4）如果不能在截止前提交作业，请写明原因。



**编程环境**

操作系统：Windows 11 家庭中文版 22H2 22621.2283 

Python编程环境：Spyder IDE 5.2.2, PyCharm 2023.1.4 (Professional Edition)





## 1. 题目

### 27638: 求二叉树的高度和叶子数目

http://cs101.openjudge.cn/practice/27638/



思路：

二叉树+递归

代码

```python
#2300012610ljx
class Tree:
    def __init__(self, left = None, right = None):
        self.left = left
        self.right = right
                   
def depth(node):
    if node == None:
        return -1
    else:
        return max(depth(node.left), depth(node.right)) + 1
    
def leave(node):
    if node == None:
        return 0
    elif node.left == None and node.right == None:
        return 1
    else:
        return leave(node.left) + leave(node.right)

n = int(input())        
nodes = [Tree() for _ in range(n)]

for i in range(n):
    l, r = map(int, input().split())
    if l != -1:
        nodes[i].left = nodes[l]
    if r != -1:
        nodes[i].right = nodes[r]

d, l = 0, 0
for i in range(n):
    d, l = max(d, depth(nodes[i])), max(l, leave(nodes[i]))
    
print(d, l)
```



代码运行截图 ==（至少包含有"Accepted"）==

<img src="C:\Users\15744\AppData\Roaming\Typora\typora-user-images\image-20240319175202488.png" alt="image-20240319175202488" style="zoom:67%;" />



### 24729: 括号嵌套树

http://cs101.openjudge.cn/practice/24729/



思路：

先创建树，再读树。

因为子节点数量不定，所以必须用某种方式记录子节点。

实在不会写看了题解，没想到可以用栈暂存父节点，天才。

代码

```python
#2300012610ljx
class Tree():
    def __init__(self, name):
        self.name = name
        self.children = []
        
def read_tree(s):
    stack = []
    node = None
    for char in s:
        if char.isalpha():
            node = Tree(char)
            if stack:
                stack[-1].children.append(node)
        elif char == '(':
            if node:
                stack.append(node)
                node = None
        elif char == ')':
            if stack:
                node = stack.pop()
    return node

def pre_order(node):
    output = [node.name]
    for child in node.children:
        output.extend(pre_order(child))
    return ''.join(output)

def post_order(node):
    output = []
    for child in node.children:
        output.extend(post_order(child))
    output.append(node.name)
    return ''.join(output)

s = input()
tree = read_tree(s)
print(pre_order(tree))
print(post_order(tree))
```



代码运行截图 ==（至少包含有"Accepted"）==

<img src="C:\Users\15744\AppData\Roaming\Typora\typora-user-images\image-20240325191208927.png" alt="image-20240325191208927" style="zoom:67%;" />



### 02775: 文件结构“图”

http://cs101.openjudge.cn/practice/02775/



思路：

建树、读树。参考了夏天明的代码。神。同样用了栈暂存父节点

代码

```python
#2300012610ljx
from sys import exit

class Node():
    def __init__(self, name):
        self.name = name
        self.dirs = []
        self.files = []
        
    def draw(self):
        text = [self.name]
        for sub_dirs in self.dirs:
            sub_text = sub_dirs.draw()
            text.extend(['|     ' + sentence for sentence in sub_text])
        for sub_files in sorted(self.files):
            text.append(sub_files)
        return text
    
numbs = 0
while True:
    numbs += 1
    stack = [Node('ROOT')]
    while (word := input()) != '*':
        if word == '#': exit(0)
        if word[0] == 'f':
            stack[-1].files.append(word)
        elif word[0] == 'd':
            stack.append(Node(word))
            stack[-2].dirs.append(stack[-1])
        else:
            stack.pop()
    print(f'DATA SET {numbs}:')
    print(*stack[0].draw(),sep='\n')
    print()
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

<img src="C:\Users\15744\AppData\Roaming\Typora\typora-user-images\image-20240326091714557.png" alt="image-20240326091714557" style="zoom:67%;" />



### 25140: 根据后序表达式建立队列表达式

http://cs101.openjudge.cn/practice/25140/



思路：

一开始不会做，后来问了AI，其实就是bfs

代码

```python
#2300012610ljx
class Node():
    def __init__(self, name):
        self.name = name
        self.left = []
        self.right = []
        
def build_tree(line):
    stack = []
    for char in line:
        node = Node(char)
        if char.isupper():
            node.right = stack.pop()
            node.left = stack.pop()
        stack.append(node)
    return stack[0]

def level_order(root):
    source = [root]
    answer = []
    while source:
        node = source.pop(0)
        answer += node.name,
        if node.left:
            source += node.left,
        if node.right:
            source += node.right,
    return answer

n = int(input())
for _ in range(n):
    line = input()
    root = build_tree(line)
    expr = level_order(root)[::-1]
    print(''.join(expr))
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==





### 24750: 根据二叉树中后序序列建树

http://cs101.openjudge.cn/practice/24750/



思路：

问了GPT。递归写得很爽

代码

```python
#2300012610ljx
class Node():
    def __init__(self, name):
        self.name = name
        self.left = []
        self.right = []
        
def build_tree(in_order, post_order):
    if not in_order or not post_order:
        return None
    
    valu = post_order[-1]
    root = Node(valu)
    divi = in_order.index(valu)
    
    in_left = in_order[:divi]
    in_right = in_order[divi+1:]

    post_left = post_order[:divi]
    post_right = post_order[divi:-1]
    
    root.left = build_tree(in_left, post_left)
    root.right = build_tree(in_right, post_right)
    
    return root

def pre_order(Node):
    if not Node:
        return []
    
    result = []
    result.append(Node.name)
    result += pre_order(Node.left)
    result += pre_order(Node.right)

    return result

in_order = input()
post_order = input()
print(*pre_order(build_tree(in_order, post_order)),sep='')
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

<img src="C:\Users\15744\AppData\Roaming\Typora\typora-user-images\image-20240326085507111.png" alt="image-20240326085507111" style="zoom:67%;" />



### 22158: 根据二叉树前中序序列建树

http://cs101.openjudge.cn/practice/22158/



思路：

改了一下前面那题的代码

代码

```python
#2300012610ljx
class Node():
    def __init__(self, name):
        self.name = name
        self.left = []
        self.right = []
        
def build_tree(pre_order, in_order):
    if not pre_order or not in_order:
        return None
    
    valu = pre_order[0]
    root = Node(valu)
    divi = in_order.index(valu)
    
    in_left = in_order[:divi]
    in_right = in_order[divi+1:]

    pre_left = pre_order[1:divi+1]
    pre_right = pre_order[divi+1:]
    
    root.left = build_tree(pre_left, in_left)
    root.right = build_tree(pre_right, in_right)
    
    return root

def post_order(Node):
    if not Node:
        return []
    
    result = []
    result += post_order(Node.left)
    result += post_order(Node.right)
    result.append(Node.name)

    return result

while True:
    try:
        pre_order = input()
        in_order = input()

        print(*post_order(build_tree(pre_order, in_order)),sep='')
    except:
        quit()
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

<img src="C:\Users\15744\AppData\Roaming\Typora\typora-user-images\image-20240326091247307.png" alt="image-20240326091247307" style="zoom:67%;" />



## 2. 学习总结和收获

==如果作业题目简单，有否额外练习题目，比如：OJ“2024spring每日选做”、CF、LeetCode、洛谷等网站题目。==

这章作业都是树，但是做得比较快，复用程度很高。

下面是普通树、二叉树的一些性质和总结。树的高度、深度、叶子数，以及bfs、dfs遍历。

二叉树有特殊的遍历，前序、中序、后序，知道任意两个表达都可重建树。

```python
class node():
    def __init__(self, name):
        self.name = name
        self.children = []
        
    def depth(self):
        if not children:
            return 0
        else:
            return max(node.depth(child) for child in self.children) + 1
    
    def leaf(self):
        if not children:
            return 1
        else:
            return sum(node.leaf(child) for child in self.children)
        
    def bfs(self):
        source = [self]
    	answer = []
    	while source:
        	node = source.pop(0)
        	answer += node.name,
     	    source += node.children
    	return answer
    
    def dfs(self):
        source = [self]
        answer = []
        while source:
            node = source.pop()
            answer += node.name,
            source += node.children[::-1]
        return answer
        
class bnode():
    def __init__(self, name):
        self.name = name
        self.left = None
        self.right = None
        
    def depth(self):
        if not self:
        	return 0
        elif not (self.left or self.right):
            return 0
        else:
            return max(bnode.depth(self.left), bnode.depth(self.right)) + 1
        
    def leaf(self):
        if not self:
            return 0
        elif not (self.left or self.right):
            return 1
        else:
            return bnode.leaf(self.left) + bnode.right(self.right)
        
    def bfs(self):
        source = [self]
    	answer = []
    	while source:
        	node = source.pop(0)
        	answer += node.name,
     	    if node.left:
        	    source += node.left,
      	    if node.right:
          	    source += node.right,
    	return answer
    
    def dfs(self):
        source = [self]
        answer = []
        while source:
			node = source.pop()
            answer += node.name
            if node.right:
                source += node.right,
            if node.left:
                source += node.left,
        return answer
    
    def pre_order(self):
        if not self:
            return []
        
        result = []
        result += self.name
        result += bnode.pre_order(self.left)
        result += bnode.pre_order(self.right)
        
        return result
    
    def in_order(self):
        if not self:
        	return []
        
        result = []
        result += bnode.in_order(self.left)
        result += self.name
        result += bnode.in_order(self.right)
        
        return result
    
    def post_order(self):
        if not self:
            return []
        
        result = []
        result += bnode.post_order(self.left)
        result += bnode.post_order(self.right)
        result += self.name
        
        return result
```



