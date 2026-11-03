# Mathematical Optimization Model to Solve Job Shop Scheduling Problem

## 1. Indices

$m \in M : \text{Index of machine}$ \
$j \in J : \text{Index of job}$

## 2. Sets

$I_{j}^{j'} \subset M : \text{Set of machines that both job } j \text{ and job } j' \text{ need to visit}$\
$T_{j} = (m_1, m_2, \dots, m_{|T_{j}|}) : \text{Trajectory, an ordered sequence of machine indices that job } j \text{ needs to visit}$

## 3. Parameters

$t_{j,m} : \text{Expected processing time of job } j \text{ on machine } m$\
$d_{j} : \text{Due date of job } j$\
$a_{j} : \text{Available time of job } j \text{ (current time if queuing, or expected completion time of its currently executing operation)}$\
$r_{m} : \text{Release time of machine } m \text{, the estimated time for the machine to complete its current operation or recover from a breakdown}$

## 4. Decision Variables

$b_{j,m} \ge 0 : \text{Start time of job } j \text{ on machine } m$\
$c_{j} \ge 0 : \text{Completion time of job } j$\
$p_{j,j',m} \in \{0, 1\} : \text{Binary precedence variable, equals } 0 \text{ if job } j \text{ precedes job } j' \text{ on machine } m\text{, and } 1 \text{ otherwise}$

## 5. Objectives and Constraints

$\text{Objectives (descending by priority in hierarchical optimization):}$

$\text{(a) Tier 1 — Minimizing cumulative tardiness:}$

$$\min \sum_{j \in J} \max\left(0, c_j - d_j\right)$$

$\text{(b) Tier 2 — Minimizing makespan:}$

$$\min \max_{j \in J} c_j$$

$\text{(c) Tier 3 — Minimizing sum of operation start times (eliminating lazy delays):}$

$$\min \sum_{j \in J} \sum_{m \in T_j} b_{j,m}$$

$\text{Subject to:}$

$(1)\ \text{Intra-job Precedence Constraints: Operations of each job must follow its trajectory } T_j\text{, where } \text{next}(m) \text{ is the machine immediately following } m \text{ in } T_j:$\
$$\qquad b_{j,m} + t_{j,m} \le b_{j,\text{next}(m)} \quad \forall j \in J,\ \forall m \in T_{j} \setminus \{m_{|T_j|}\}$$

$(2)\ \text{Machine Release and Job Availability Constraints:}$\
$$\qquad b_{j,m} \ge r_{m} \quad \forall j \in J,\ \forall m \in T_{j}$$
$$\qquad b_{j,m_1} \ge a_{j} \quad \forall j \in J$$

$(3)\ \text{Disjunctive Machine Constraints: Each machine processes at most one job at a time (with large constant } \mathcal{M}\text{):}$\
$$\qquad b_{j,m} + t_{j,m} - \mathcal{M} \cdot p_{j,j',m} \le b_{j',m} \quad \forall j, j' \in J,\ \forall m \in I_{j}^{j'}$$
$$\qquad b_{j',m} + t_{j',m} - \mathcal{M} \cdot \left(1 - p_{j,j',m}\right) \le b_{j,m} \quad \forall j, j' \in J,\ \forall m \in I_{j}^{j'}$$

$(4)\ \text{Completion Time Constraint:}$\
$$\qquad c_{j} = b_{j,m_{|T_j|}} + t_{j,m_{|T_j|}} \quad \forall j \in J$$

