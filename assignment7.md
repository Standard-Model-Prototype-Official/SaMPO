# Assignment #7: April 月考

Updated 1557 GMT+8 Apr 3, 2024

2024 spring, Complied by ==罗景轩，地空==



**说明：**

1）请把每个题目解题思路（可选），源码Python, 或者C++（已经在Codeforces/Openjudge上AC），截图（包含Accepted），填写到下面作业模版中（推荐使用 typora https://typoraio.cn ，或者用word）。AC 或者没有AC，都请标上每个题目大致花费时间。

2）提交时候先提交pdf文件，再把md或者doc文件上传到右侧“作业评论”。Canvas需要有同学清晰头像、提交文件有pdf、"作业评论"区有上传的md或者doc附件。

3）如果不能在截止前提交作业，请写明原因。



**编程环境**

操作系统：Windows 11 家庭中文版 22H2 22621.2283 

Python编程环境：Spyder IDE 5.2.2, PyCharm 2023.1.4 (Professional Edition)



## 1. 题目

### 27706: 逐词倒放

http://cs101.openjudge.cn/practice/27706/



思路：

栈

代码

```python
line = input()
stack = []
word = ''
for char in line:
    if char == ' ':
        stack.append(word)
        word = ''
    else:
        word += char
stack.append(word)        
print(*reversed(stack))
```



代码运行截图 ==（至少包含有"Accepted"）==

<img src="C:\Users\15744\AppData\Roaming\Typora\typora-user-images\image-20240407102203608.png" alt="image-20240407102203608" style="zoom:67%;" />



### 27951: 机器翻译

http://cs101.openjudge.cn/practice/27951/



思路：

队列

代码

```python
from collections import deque

n, m = map(int, input().split())
line = list(map(int, input().split()))

memo = deque()
chec = 0

for word in line:
    if word not in memo:
        if len(memo) == n:
            memo.popleft()
        memo.append(word)
        chec += 1
        
print(chec)

```



代码运行截图 ==（至少包含有"Accepted"）==

<img src="C:\Users\15744\AppData\Roaming\Typora\typora-user-images\image-20240407103404186.png" alt="image-20240407103404186" style="zoom:67%;" />



### 27932: Less or Equal

http://cs101.openjudge.cn/practice/27932/



思路：

排序

代码

```python
n, k = map(int, input().split())

line = list(map(int, input().split()))
line.sort()

if k == 0:
    x = 1 if line[0] > 1 else -1
elif k == n:
    x = line[-1]
else:
    x = line[k-1] if line[k-1] != line[k] else -1
        
print(x)
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

<img src="C:\Users\15744\AppData\Roaming\Typora\typora-user-images\image-20240407104952635.png" alt="image-20240407104952635" style="zoom:67%;" />



### 27948: FBI树

http://cs101.openjudge.cn/practice/27948/



思路：

树

代码

```python
#2300012610ljx

def FBItree(s):
    if '0' in s and '1' in s:
        node = 'F'
    elif '1' in s:
        node = 'I'
    else:
        node = 'B'
        
    if len(s) > 1:
        mid = len(s) // 2
        left = FBItree(s[:mid])
        right = FBItree(s[mid:])
        return left + right + node
    else:
        return node
    
N = int(input())
s = input()
print(FBItree(s)
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

<img src="C:\Users\15744\AppData\Roaming\Typora\typora-user-images\image-20240407112609189.png" alt="image-20240407112609189" style="zoom:67%;" />



### 27925: 小组队列

http://cs101.openjudge.cn/practice/27925/



思路：

字典+队列。用了队列保存索引

代码

```python
#2300012610ljx

from collections import deque

t = int(input())
teams = {i: deque(map(int, input().split())) for i in range(t)}
index = deque()
group = {i: deque() for i in range(t)}

while True:
    command = input().split()
    if command[0] == 'STOP':
        break
    elif command[0] == 'ENQUEUE':
        person = int(command[1])
        for i in range(t):
            if person in teams[i]:
                group[i].append(person)
                if i not in index:
                    index.append(i)
                break
    elif command[0] == 'DEQUEUE':
        group_num = index[0]
        print(group[group_num].popleft())
        if not group[group_num]:
            index.popleft()

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

<img src="C:\Users\15744\AppData\Roaming\Typora\typora-user-images\image-20240407115852147.png" alt="image-20240407115852147" style="zoom:67%;" />



### 27928: 遍历树

http://cs101.openjudge.cn/practice/27928/



思路：

字典+递归

代码

```python
#2300012610ljx
from collections import defaultdict

n = int(input())
index = defaultdict(list)
parents = []
children = []

for _ in range(n):
    line = list(map(int, input().split()))
    parents += line[0],
    if len(line) > 1:
        child = line[1:]
        children += child
        index[line[0]] += child
        
def traversal(node):
    seq = sorted(index[node] + [node])
    for i in seq:
        if i == node:
            print(node)
        else:
            traversal(i)
            
traversal((set(parents)-set(children)).pop())

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

<img src="C:\Users\15744\AppData\Roaming\Typora\typora-user-images\image-20240407122817748.png" alt="image-20240407122817748" style="zoom:67%;" />



## 2. 学习总结和收获

==如果作业题目简单，有否额外练习题目，比如：OJ“2024spring每日选做”、CF、LeetCode、洛谷等网站题目。==

这周意外地忙，所以在周天补课的时候才补了作业。

前四题都挺简单的，有点像计概的题。第五题开始变难，五六两题都有数据结构的嵌套。

第五题有多个队列在排队，所以队列本身是大队列的元素，用了索引记录大队列、用字典记录小队列的元素。

第六题比较不好理解，是一种新的遍历方式，但是用递归比较好实现。建树的方式并没有用class，而是用字典逃课了，也比较简单。值得注意的是要找到根。



感觉计概和数算各有各的难处，计概偏重语法以及比较复杂的处理，数算则是从典型的案例出发，用几个数据结构解决大部分的问题，复用更为频繁。希望以后能够扎实这方面。



