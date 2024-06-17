# Assignment #3: March月考

Updated 1537 GMT+8 March 6, 2024

2024 spring, Complied by ==罗景轩，地空==



**说明：**

1）The complete process to learn DSA from scratch can be broken into 4 parts:
- Learn about Time and Space complexities
- Learn the basics of individual Data Structures
- Learn the basics of Algorithms
- Practice Problems on DSA

2）请把每个题目解题思路（可选），源码Python, 或者C++（已经在Codeforces/Openjudge上AC），截图（包含Accepted），填写到下面作业模版中（推荐使用 typora https://typoraio.cn ，或者用word）。AC 或者没有AC，都请标上每个题目大致花费时间。

3）提交时候先提交pdf文件，再把md或者doc文件上传到右侧“作业评论”。Canvas需要有同学清晰头像、提交文件有pdf、"作业评论"区有上传的md或者doc附件。

4）如果不能在截止前提交作业，请写明原因。



**编程环境**

操作系统：Windows 11 家庭中文版 22H2 22621.2283 

Python编程环境：Spyder IDE 5.2.2, PyCharm 2023.1.4 (Professional Edition)





## 1. 题目

**02945: 拦截导弹**

http://cs101.openjudge.cn/practice/02945/



思路：

很典的dp。参考了一位同学的代码，每当可以拦截的时候，更新数组。

##### 代码

```python
#2300012610ljx
n = int(input())
arr = [*map(int, input().split())]
dp = [1 for _ in range(n)]
for i in range(1, n):
    for j in range(i):
        if arr[i] <= arr[j]:
            dp[i] = max(dp[j] + 1, dp[i])
#            print(dp)
print(max(dp))
```



代码运行截图 ==（至少包含有"Accepted"）==

<img src="C:\Users\15744\AppData\Roaming\Typora\typora-user-images\image-20240306212416657.png" alt="image-20240306212416657" style="zoom:67%;" />



**04147:汉诺塔问题(Tower of Hanoi)**

http://cs101.openjudge.cn/practice/04147



思路：

递归，具体在后面细说。代码可以非常简单

##### 代码

```python
#2300012610ljx
def m(a, b, n):
    print(n,":"+a+"->"+b, sep='')
    
def H(a, b, c, n):
    if n == 1:
        m(a, c, 1)
    else:
        H(a, c, b, n-1)
        m(a, c, n)
        H(b, a, c, n-1)
        
n, a, b, c = input().split()       
H(a, b, c, int(n))

```



代码运行截图 ==（至少包含有"Accepted"）==

<img src="C:\Users\15744\AppData\Roaming\Typora\typora-user-images\image-20240308224530796.png" alt="image-20240308224530796" style="zoom:67%;" />



**03253: 约瑟夫问题No.2**

http://cs101.openjudge.cn/practice/03253



思路：

要求的模拟实现，没什么好想的，主要是练习deque和注意调参数

##### 代码

```python
#2300012610ljx
from collections import deque

while True:
    n, p, m = map(int, input().split())
    if n == 0 and p == 0 and m == 0:
        break
    else:
        ans = []
        arr = deque(range(n))
        arr.rotate(-p+1)
        for _ in range(n):
            arr.rotate(-m+1)
            ans.append(str(arr.popleft() + 1))
        print(','.join(ans))
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

<img src="C:\Users\15744\AppData\Roaming\Typora\typora-user-images\image-20240308231345432.png" alt="image-20240308231345432" style="zoom:67%;" />



**21554:排队做实验 (greedy)v0.2**

http://cs101.openjudge.cn/practice/21554



思路：

贪心加模拟

##### 代码

```python
#2300012610ljx
import heapq

n = int(input())
arr = input().split()
ans = []
t1 = 0
t2 = 0

for i in range(n):
    arr[i] = [int(arr[i]), i + 1]
    
heapq.heapify(arr)
for _ in range(n):
    add = heapq.heappop(arr)
    ans += add[1],
    t2 += t1
    t1 += add[0]

print(*ans)
print("%.2f"%(t2/n))

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

<img src="C:\Users\15744\AppData\Roaming\Typora\typora-user-images\image-20240308233934061.png" alt="image-20240308233934061" style="zoom:67%;" />



**19963:买学区房**

http://cs101.openjudge.cn/practice/19963



思路：

比较麻烦的实现。搞清楚就不难了！

##### 代码

```python
#2300012610ljx
def zhongweishu(n, L):
    L=sorted(L)
    if n%2 == 0:
        return((L[n//2]+L[n//2-1])/2)
    else:
        return(L[(n-1)//2])
    
n = int(input())    
pairs = [i[1:-1] for i in input().split()]
distances = [sum(map(int,i.split(','))) for i in pairs]
price = list(map(int, input().split()))
rates = [0]*n

for i in range(n):
    rates[i]=distances[i]/price[i]
        
mr = zhongweishu(n, rates)
mp = zhongweishu(n, price)
cnt = 0

for i in range(n):
    if rates[i]>mr and price[i]<mp:
        cnt+=1
        
print(cnt)
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

<img src="C:\Users\15744\AppData\Roaming\Typora\typora-user-images\image-20240308234631849.png" alt="image-20240308234631849" style="zoom:67%;" />



**27300: 模型整理**

http://cs101.openjudge.cn/practice/27300



思路：

排序，输出。做得比较繁琐，希望学学大佬简洁的写法。

##### 代码

```python
#2300012610ljx
def s(a):
    if a == "M":
        return 0
    else:
        return 1
    
def inpu(st):
    a, b = st.split("-")
    c, d = b[:-1], b[-1]
    return (a, c, d)

arr = []
n = int(input())
for _ in range(n):
    arr += inpu(input()),

arr = sorted(arr, key = lambda x: (x[0], s(x[2]), float(x[1])))

nam = arr[0][0]
ans = [[nam]]
j = 0

for i in arr:
    if i[0] != nam:
        nam = i[0]
        ans += [[nam]]
        j += 1
    ans[j] += i[1] + i[2],

for i in ans:
    print(i[0]+": "+", ".join(i[1:]))
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

<img src="C:\Users\15744\AppData\Roaming\Typora\typora-user-images\image-20240309005832008.png" alt="image-20240309005832008" style="zoom:67%;" />。



## 2. 学习总结和收获

==如果作业题目简单，有否额外练习题目，比如：OJ“2024spring每日选做”、CF、LeetCode、洛谷等网站题目。==

第一题是dp。和每一道dp一样，难点都在状态转移方程。

汉诺塔用了递归，想清楚了就很快。值得注意的是要记得把函数中的各个参数都搞清楚。比如我的代码里H函数有三加一个变量，前三个分别表示待转移的位置，空的位置和目的的位置，在递推时不要搞晕。m函数则是打印出步骤的函数。挪动一个n层塔由a至c，相当于挪动一个n-1层塔由a至b，挪动底层由a至c，再挪动一个n-1层塔由b至c。然后就AC了。

约瑟夫问题用了deque的两个函数，rotate和popleft，注意参数。究竟是p还是-p还是-p+1还是-p-1，费了很久才试出来。

排队做实验是贪心，时间越短就越靠前。但是由于要求输出顺序，所以将题目中的数据处理成了时间加序号的数组，然后用堆，每次弹出最小值。时间是累进的，用了一个小技巧，花了两个参数输出累加，很方便，建议推广。

买学区房比较繁琐，但是很简单。

模型整理则是数据处理方面比较困难。为此，我定义了两个函数，其中一个函数用于比较M和B，另一个函数用来将输入的字符串分为三个部分，排序则是要调整好参数，这里是`key = lambda x: (x[0], s(x[2]), float(x[1]))`。为了区分不同的名称以分开输入，构建了二维数组，并且判断名称是否一致，最后输出。代码比较繁琐，希望能找到更简洁的写法。

*总结

感觉collection库实在是个宝，很多数据结构都可以直接用。

