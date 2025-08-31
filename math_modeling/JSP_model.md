# Mathematical Optimization Model to Solve Job Shop Scheduling Problem

## 1. Index

$m \in M : \text{Index of machine}$ \
$j \in J : \text{Index of job}$

## 2. Set

$I_{j}^{j'} \subset J : \text{A set of machines that job } j \text{ and } j' \text{ both need to visit}$\
$T_{j} = (m_1, m_2, ..., m_{\mid T_{j}\mid} ) : \text{Trajectory, an ordered set containing machine indices that job } j \text{ needs to visit}$\

## 3. Paremeters

$t_{j,m} : \text{Expected processing time of job } j \text{ on machine } m $\
$d_{j} : \text{Due time of job } j $\
$r_{m} : \text{Release time of machine } m \text{, the estimated time for the machine to complete its current operation or recover from a breakdown, calculated from the start of the current scheduling cycle}$

## 4. Decision Variables

$b_{j,m} : \text{Time that job } j \text{ started to be processed on machine } m$\
$c_{j} : \text{Completion time of job } j $\
$p_{j,j',m} : \text{Precedence variable, equals 0 if job } j \text{ is processed on machine } m \text{ before job } j' $

## 5. Objective and Constraints

$\text{Objectives (decending by priority in hierarchical optimization):} $

$\text{(a) Minimizing cumulative tardiness:} $

$\min \displaystyle \sum_{j \in J}\left({c_j - d_j} \right)$

$\text{(b) Minimizing makespan:} $

$\min \max \ c_j \quad \forall j\in J $

$\text{s.t.}$

$(1)\ \text{Precedence Constraints: All operations must be processed following pre-defined trajectory. Where } next(m) \text{ refers to the index of the machine immediately following machine m in } T_{j} $\
$\qquad b_{j,m} + t_{j,m} <= b_{j,next(m)} \quad \forall j \in J, m \in T_{j} $

$(2)\ \text{Machine release constraint: The operations on a machine must commence after the machine release time} $\
$\qquad b_{j,m} >= r_{m} \quad \forall j \in J, m \in T_{j} $

$(3)\ \text{Disjunctive constraints: One machine can process at most one job at a time: }$\

$\qquad \left( b_{j,m} + t_{j,m} \right) \times \left( 1 - p_{j,j',m} \right) <= b_{j,m^{'}} \quad \forall j, j' \in J, m \in I_{j}^{j'}$

$\qquad \left( b_{j^{'}m} + t_{j^{'},m} \right) \times p_{j,j',m} <= b_{j,m} \quad \forall j, j' \in J, m \in I_{j}^{j'}$

$(4)\ \text{Completion time constraint: }$\
$\qquad c_{j} = b_{j,T_{\mid T_{j}\mid}} + t_{j,T_{\mid T_{j}\mid}} \quad \forall j \in J$
