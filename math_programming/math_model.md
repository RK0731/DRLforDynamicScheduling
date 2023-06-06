## Mathematical Optimization Model to Solve Job Shop Scheduling Problem

### 1. Index

$
m \in M : \text{Index of machine}\\
j \in J : \text{Index of job}\\
$

### 2. Set

$
J_{m} \subset J : \text{A set of jobs that need to be processed by machine } m \\
T_{j} : \text{Trajectory of job } j \text{, a sequence of the index of machines that the job needs to visit}\\
$

### 3. Paremeters

$
t_{jm} : \text{Expected processing time of job } j \text{ on machine } m \\
d_{j} : \text{Due time of job } j \\
r_{m} : \text{Release time of machine } m \text{, the expected time that machine complete current operation or restore from breakdown}
$

### 4. Decision Variables

$
p_{jj'm} : \text{Precedence variable} \text{, equals 1 if job } j \text{ is processed on machine } m \text{ before job } j'\\
s_{jm} : \text{Time variable, the time when machine } m \text{ starts processing job } j \\
c_{j} : \text{Completion time of job } j \\
$

### 5. Objective and Constraints
$\text{Alternative objectives:} \\[10pt]
\text{(a) Minimizing cumulative tardiness:} \\[10pt]
\min \displaystyle \sum_{j \in J}\left({c_j - d_j} \right) \\[10pt]
\text{(b) Minimizing makespan (for static problem instance):} \\[10pt]
\min \max \ c_j \quad \forall j\in J\\[10pt]
\text{s.t.}\\
(1)\ \text{All operations must be processed by pre-defined sequence} \\
(2)\ \text{The operations of jobs on a mahcine must conform their precedence}\\
\text{Dummy constraint: get the tardiness of a job}
$