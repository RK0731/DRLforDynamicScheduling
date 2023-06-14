## Mathematical Optimization Model to Solve Job Shop Scheduling Problem

### 1. Index

$m \in M : \text{Index of machine}$ \
$j \in J : \text{Index of job}$

### 2. Set

$I_{j}^{j'} \subset J : \text{A set of machines that job } j \text{ and } j' \text{ both need to visit}$\
$T_{j} = \{m_1, m_2, ..., m_{\mid T_{j}\mid} \} : \text{Trajectory, an ordered set containing machine indices that job } j \text{ needs to visit}$\
$A_{j}^{m} \subset M : \text{A set of machines that job } j \text{ needs to visit after visiting machine } m$

### 3. Paremeters

$t_{jm} : \text{Expected processing time of job } j \text{ on machine } m $\
$d_{j} : \text{Due time of job } j $\
$r_{m} : \text{Release time of machine } m \text{, the expected time that machine complete current operation or restore from breakdown}$

### 4. Decision Variables

$b_{jm} : \text{Operation begin time variable} \text{, time that job } j \text{ started to be processed on machine } m $\
$c_{j} : \text{Completion time of job } j $\
$p_{jj'm} : \text{Precedence variable} \text{, equals 0 if job } j \text{ is processed on machine } m \text{ before job } j' $

### 5. Objective and Constraints
$\text{Objectives (decending by priority in hierarchical optimization):} $

$\text{(a) Minimizing cumulative tardiness:} $

$\min \displaystyle \sum_{j \in J}\left({c_j - d_j} \right)$

$\text{(b) Minimizing makespan:} $

$\min \max \ c_j \quad \forall j\in J $

$\text{s.t.}$

$(1)\ \text{All operations must be processed following pre-defined trajectory} $\
$\qquad b_{jm} + t_{jm} <= b_{jm^{'}} \quad \forall j \in J, m \in T_{j}, m^{'} \in A_{j}^{m} $

$(2)\ \text{The operations of jobs on a mahcine must commence after the machine released} $\
$\qquad b_{jm} >= r_{m} \quad \forall j \in J, m \in T_{j} $

$(3)\ \text{Precedence 1, let job } j \text{ prcedes job } j' \text{ on machine } m $\
$\qquad \left( b_{jm} + t_{jm} \right) \times \left( 1 - p_{jj'm} \right) <= b_{jm^{'}} \quad \forall j, j' \in J, m \in I_{j}^{j'}$

$(4)\ \text{Precedence 2, let job } j' \text{ prcedes job } j \text{ on machine } m $\
$\qquad \left( b_{j^{'}m} + t_{j^{'}m} \right) \times p_{jj'm} <= b_{jm} \quad \forall j, j' \in J, m \in I_{j}^{j'}$

$(5)\ \text{Calculate completion time }$\
$\qquad c_{j} = b_{jm_{\mid T_{j}\mid}} + t_{jm_{\mid T_{j}\mid}} \quad \forall j \in J$