# Assignment #4: 排序、栈、队列和树

Updated 0005 GMT+8 March 11, 2024

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

### 05902: 双端队列

http://cs101.openjudge.cn/practice/05902/



思路：

deque

代码

```python
#2300012610ljx
from collections import deque

for _ in range(int(input())):
    arr = deque([])
    for _ in range(int(input())):
        fac, obj = map(int, input().split())
        if fac == 1:
            arr.append(obj)
        else:
            if obj == 0:
                arr.popleft()
            else:
                arr.pop()
    if arr:
        print(*list(arr))
    else:
        print("NULL")
```



代码运行截图 ==（至少包含有"Accepted"）==

<img src="C:\Users\15744\Pictures\Screenshots\屏幕截图 2024-03-12 165455.png" alt="屏幕截图 2024-03-12 165455" style="zoom:67%;" />



### 02694: 波兰表达式

http://cs101.openjudge.cn/practice/02694/



思路：

栈

代码

```python
#罗景轩2300012610
express= list(input().split())
def compute(a, b, c):
    if a =="+":
        return(b + c)
    elif a =="-":
        return(b - c)
    elif a =="*":
        return(b * c)
    else:
        return(b / c)

PMTD = ["+","-","*","/"]
nums = []

for i in range(len(express)):
    if express[-i-1] in PMTD:
        a = float(nums.pop())
        b = float(nums.pop())
        nums.append(compute(express[-i-1], a, b))
    else:
        nums.append(float(express[-i-1]))

print("{:.6f}".format(nums[0]))

```



代码运行截图 ==（至少包含有"Accepted"）==

<img src="C:\Users\15744\Pictures\Screenshots\屏幕截图 2024-03-12 165634.png" alt="屏幕截图 2024-03-12 165634" style="zoom:67%;" />



### 24591: 中序表达式转后序表达式

http://cs101.openjudge.cn/practice/24591/



思路：

栈，模板题

代码

```python
#2300012610ljx
def answ(stri):

    PMTD = {'+':1, '-':1, '*':2, '/':2}
    stack, ans = [], []
    number = ''

    for char in stri:
        if char.isnumeric() or char == '.':
            number += char
        else:
            if number:
                num = float(number)
                ans.append(int(num) if num.is_integer() else num)
                number = ''
                
            if char in '+-*/':
                while stack and stack[-1] in '+-*/'\
                and PMTD[char] <= PMTD[stack[-1]]:
                    ans.append(stack.pop())
                stack.append(char)
            elif char == '(':
                stack.append(char)
            elif char == ')':
                while stack and stack[-1] != '(':
                    ans.append(stack.pop())
                stack.pop()
        #print(stack, ans, char, number)
                
    if number:
        num = float(number)
        ans.append(int(num) if num.is_integer() else num)
        
    while stack:
        ans.append(stack.pop())
        
    return ans

for _ in range(int(input())):
    print(*answ(input()))

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

<img src="C:\Users\15744\Pictures\Screenshots\屏幕截图 2024-03-12 180132.png" alt="屏幕截图 2024-03-12 180132" style="zoom:67%;" />



### 22068: 合法出栈序列

http://cs101.openjudge.cn/practice/22068/



思路：

栈，模板题

代码

```python
#2300012610ljx
def if_stack(sub, obj):
    stack = []
    sub = list(sub)
    
    for char in obj:
        while (not stack or stack[-1] != char) and sub:
            stack.append(sub.pop(0))
            #print(obj, stack)
        if not stack or stack[-1] != char:
            return False
        
        stack.pop()
        
    return True
    
tar = input()
while True:
    try:
        acc = input()
        if len(acc) == len(tar):
            print('YES' if if_stack(tar, acc) else 'NO')
        else:
            print('NO')
    except:
        break                
   

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

<img src="C:\Users\15744\AppData\Roaming\Typora\typora-user-images\image-20240312201417850.png" alt="image-20240312201417850" style="zoom:67%;" />



### 06646: 二叉树的深度

http://cs101.openjudge.cn/practice/06646/



思路：

树

代码

```python
#2300012610ljx
class tree():
    def __init__(self):
        self.left = None
        self.right = None

def module(node):
    if node is None:
        return 0
    return max(module(node.left), module(node.right)) + 1

n = int(input())
nodes = [tree() for _ in range(n)]

for i in range(n):
    l, r = map(int, input().split())
    if l != -1:
        nodes[i].left = nodes[l-1]
    if r != -1:
        nodes[i].right = nodes[r-1]   

print(module(nodes[0]))
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![image-20240312203952110](C:\Users\15744\AppData\Roaming\Typora\typora-user-images\image-20240312203952110.png)



### 02299: Ultra-QuickSort

http://cs101.openjudge.cn/practice/02299/



思路：

合并排序+双指针+分治。学到了很多

代码

```python
#2300012610ljx
def merge_count(arr, l, r):
    #print(l, r)
    if l >= r:
        return 0
    
    mid = (l + r) // 2
    count = merge_count(arr, l, mid) + merge_count(arr, mid + 1, r)
    
    temp = []
    i, j = l, mid + 1
    while i <= mid and j <= r:
        if arr[i] <= arr [j]:
            temp.append(arr[i])
            i += 1
        else:
            temp.append(arr[j])
            j += 1
            count += (mid - i + 1)                
    
    while i <= mid:
        temp.append(arr[i])
        i += 1
        
    while j <= r:
        temp.append(arr[j])
        j += 1
        
    for i in range(len(temp)):
        arr[l + i] = temp[i]
        
    #print(temp)
    return count

while True:
    n = int(input())
    if n != 0:
        ans = []
        for _ in range(n):
            ans += int(input()),
        print(merge_count(ans, 0, n-1))
    else:
        break
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

<img src="C:\Users\15744\AppData\Roaming\Typora\typora-user-images\image-20240312235813326.png" alt="image-20240312235813326" style="zoom:67%;" />



## 2. 学习总结和收获

==如果作业题目简单，有否额外练习题目，比如：OJ“2024spring每日选做”、CF、LeetCode、洛谷等网站题目。==

这次作业难度上来了，但也学到了很多。

栈，队列，双向队列，树。最后还有归并排序(merge).

[链表 - OI Wiki (oi-wiki.org)](https://oi-wiki.org/ds/linked-list/)这个网站挺好用的，很全，但是代码还需要补充。

#### 栈stack

```python
class stack:
    def __init__(self):
        self.items = []#用列表实现类
        
    def is_empty(self):
        return self.items == []#判断是否为空
    
    def push(self, item):
        self.items.append(item)#添加数据
        
    def pop(self):
        return self.items.pop()#弹出数据
        
    def peek(self):
        return self.items[len(self.items)-1]#查看数据
    
    def size(self):
        return len(self.items)#栈长度
```

#### 队列queue

```python
class queue:
    def __init__(self):
        self.items = []#用列表实现类
        
    def is_empty(self):
        return self.items == []#判断是否为空
    
    def enqueue(self, item):
        self.items.insert(0, item)#添加数据
        
    def dequeue(self):
        return self.items.pop()#弹出数据
        
    def size(self):
        return len(self.items)#队列长度
```

### 双端队列deque

```python
class deque:
    def __init__(self):
        self.items = []#用列表实现类
        
    def is_empty(self):
        return self.items == []#判断是否为空
    
    def addFront(self, item):
        self.items.append(item)#添加数据
        
    def addRear(self, item):
        self.items.insert(0, item)#添加数据
        
    def removeFront(self):
        return self.items.pop()#弹出数据
        
    def removeRear(self):
        return self.items.pop(0)#弹出数据
            
    def size(self):
        return len(self.items)#双端队列长度
```

### 单向链表linkedList

```python
class node:#定义节点
    def __init__(self, value):#节点由一个数据和指针组成
        self.value = value#数据
        self.next = None#指针
        
class linkedList:#定义链接
    def __init__(self):
        self.head = None#链表的头部
        
    def insert(self, value):
        new_node = node(value)#构造新节点
        if self.head is None:#如果链表内没有节点
            self.head = new_node#那么新节点就是链表的头部
        else:#否则
            current = self.head#从头开始
            while current.next:#有下一个的时候
                current = current.next#就挪到后一个节点
            current.next = new_node#直到最后一个节点，它的next，就是新节点
            
    def delete(self, value):#删除节点
        if self.head is None:
            return#没节点你还删个啥
        
        if self.head.value == value:#如果头部节点的值符合
            self.head = self.head.next#头部就改为下一个
        else:#否则
            current = self.head#从头开始
            while current.next:#有后一个的时候
                if current.next.value == value:#如果值符合
                    current.next = current.next.next#就把指向改掉
                    break
                current = current.next#不符合就改到下一个
                
    def display(self):#打印
        current = self.head#从头开始
        while current:
            print(current.value, end=' ')#印一下
            current = current.next#印完改成下一个
        print()
        
	#下面是一些别的函数
    def reverse(self):
        prev = None
        curr = self.head
        while curr:
            next_node = curr.next
            curr.next = prev
            prev = curr
            curr = next_node
        self.head = prev
        
    def reverse_another(self):
        curr = self.head.next
        self.head.next = None
        while curr is not None:
            curr_another = curr
            curr = curr.next
            curr_another.next = self.head
            self.head =curr_another
        
```

### 双向链表DoublelinkedList

```python
class node:
    def __init__(self, value):
        self.value = value
        self.prev = None
        self.next = None
        
class DoublelinkedList:
    def __init__(self):
        self.head = None
        self.tail = None
        
    def insert_before(self, node, new_node):
        if node is None:
            self.head = new_node
            self.tail = new_node
        else:
            new_node.next = node
            new_node.prev = node.prev
            if node.prev is not None:
                node.prev.next = new_node
            else:
                self.head = new_node
            node.prev = new_node
            
    def insert_after(self, node, new_node):
        if node is None:
            self.head = new_node
            self.tail = new_node
        else:
            new_node.prev = node
            new_node.next = node.next
           	if node.next is not None:
                node.next.prev = new_node
            else:
                self.tail = new_node
            node.prev = new_node
     
    def delete(self, value):
        if self.head == None:
            return
        
        if self.head.value == value:
            self.head = self.head.next
            self.head.prev = None
        else:
            current = self.head
            while current.next:
                if current.next.value == value:
                    current.next = current.next.next
                    current.next.prev = current
          
    def display_forward(self):
        current = self.head
        while current is not None:
            print(current.value, end =' ')
            current = current.next
        print()
        
    def display_backward(self):
        current = self.tail
        while current is not None:
            print(current.value, end =' ')
            current = current.prev
        print()
```

### 单向循环链表

```python
class Node:
    def __init__(self, data, next=None):
        self.data = data
        self.next = next
        
class circlinklist:
    def __init__(self):
        self.tail = None
        self.size = 0
    
    def isEmpty(self):
        return self.size == 0
    
    def pushFront(self, data):
        node = Node(data)
        if self.tail == None:
            self.tail == node
            node.next = self.tail
        else:
            node.next = self.tail.next
            self.tail.next == node
        self.size += 1
        
    def pushBack(self, data):
        self.pushFront(data)
        self.tail = self.tail.next
        
    def popFront(self):
        if isEmpty(self):
            return None
        else:
            node = self.tail.next
            self.size -= 1
            if isEmpty(self):
            	self.tail = None
            else:
                self.tail.next = node.next
        return node.data
    
    def remove(self, data):
        if isEmpty(self):
            return None
        else:
            end = self.tail
            while end.next.data != data:
                end = end.next
                if end == self.tail:
                    return False
            self.size -= 1
            if end.next == self.tail:
                self.tail = end
            end.next = end.next.next
            return True
        
    def print_list(self):
        if not isEmpty:
            curr = self.tail.next
            while True:
                print(curr.data, end=' ')
                if curr == self.tail:
                    break
                curr = curr.next
            print()
```

双向循环链表和这些都类似就不写了。

### 二叉树

```python
class binaryTree:
	def __init__(self, root):
        self.key = root
        self.left = None
        self.right = None
        
    def insertleft(self, new_node):
        if self.left = None:
            self.left = binaryTree(new_node)
        else:
            curr = binaryTree(new_node)
            curr.left = self.left
            self.left = curr
            
    def insertright(self, new_node):
        if self.right = None:
            self.right = binaryTree(new_node)
        else:
            curr = binaryTree(new_node)
            curr.right = self.right
            self.right = curr
            
    def get_left(self):
        return self.left
            
	def get_right(self):
        return self.right
    
    def set_root(self, data):
        self.key = data
        
    def get_root(self):
        return self.key

```

### 归并排序

```python
def merge_count(arr, l, r):
    if l >= r:
        return 0
    
    mid = (l + r) // 2
    count = merge_count(arr, l, mid) + merge_count(arr, mid + 1, r)
    
    temp = []
    i, j = l, mid + 1
    while i <= mid and j <= r:
        if arr[i] <= arr [j]:
            temp.append(arr[i])
            i += 1
        else:
            temp.append(arr[j])
            j += 1
            count += (mid - i + 1)                
    
    while i <= mid:
        temp.append(arr[i])
        i += 1
        
    while j <= r:
        temp.append(arr[j])
        j += 1
        
    for i in range(len(temp)):
        arr[l + i] = temp[i]
        
    return count
```

链表的实现大量运用了指针，学习的时候常常感慨怎么有人能聪明到这种程度。比如reverse的两种写法，用了三到四个指针轮流互相赋值，简直就是艺术，



双向链表的delete函数没找到，是自己写的，有待检验。

树是递归的艺术，但是自相似性其实在别的数据结构里面比比皆是，只是因为过于平凡被忽略了。栈也可以认为是最深处的数据和浅部的栈，链表也可以认为是head和子链表。







