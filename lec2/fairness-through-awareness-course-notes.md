# Lecture 2: Fairness Through Awareness

## Individual fairness, statistical parity, and the geometry of fair classification

**Primary reading:** Cynthia Dwork, Moritz Hardt, Toniann Pitassi, Omer Reingold, and Richard Zemel, [“Fairness Through Awareness”](https://arxiv.org/pdf/1104.3913), arXiv:1104.3913v2 (2011).

**Format:** Self-contained course-note chapter. Page references below refer to the PDF page numbers. The paper is primarily theoretical: its central results are definitions, optimization formulations, characterizations, and bounds rather than empirical benchmark results.

## Learning objectives

After studying these notes, you should be able to:

1. Explain why removing a protected attribute does not guarantee fair treatment.
2. State the paper's individual-fairness principle: similar individuals should receive similar distributions over outcomes.
3. Formalize that principle as a Lipschitz condition between a task-specific metric and a probability-distribution metric.
4. Explain why the classifier is randomized and why fairness constrains distributions rather than only realized outcomes.
5. Write and interpret the Fairness LP, including its utility objective and pairwise fairness constraints.
6. Compare total variation distance and the relative $\ell_\infty$ distance used in the paper.
7. Explain why statistical parity is useful but insufficient as a complete fairness criterion.
8. Use Earthmover distance to characterize when individual fairness implies group-level statistical parity.
9. Explain the paper's two-stage construction for “fair affirmative action.”
10. Relate the framework to differential privacy and the exponential mechanism.
11. Interpret the role of doubling dimension in the paper's fairness–utility bounds.
12. Identify the framework's central unresolved assumption: the construction and legitimacy of the similarity metric.

---

## 1. The problem the paper is trying to solve

Classification allocates consequential outcomes. A university admits or rejects applicants. A lender presents different credit products. An advertising network decides which opportunity to show. In each case, the decision maker wants utility, but society also wants protection against discrimination.

This motivation and the paper's overview appear in §1 (pp. 1–4).

A tempting response is **fairness through blindness**: delete race, sex, disability status, or another protected attribute and then optimize as usual. The paper rejects this as inadequate. Rich behavioral data can encode protected status redundantly. ZIP code, browsing history, social connections, purchases, and many other features may jointly reconstruct information that was explicitly removed.

The paper starts from a different question:

> Which differences between people are relevant to this task, and how much difference in treatment may those relevant differences justify?

The answer is represented by a task-specific distance function over individuals. If two people are close according to that function, the classifier must treat them similarly. If they are far apart, the classifier has more freedom.

This is **fairness through awareness**. The system does not pretend differences are absent. It becomes explicitly aware of which differences are legitimate for the task and constrains treatment accordingly.

### 1.1 Blindness versus awareness

| Approach | Core rule | What it can miss |
|---|---|---|
| Fairness through blindness | Do not directly use protected attributes | Other variables may redundantly encode them; unequal treatment can survive feature deletion |
| Statistical parity alone | Equalize group-level outcome distributions | A decision maker can choose the “wrong” individuals inside a group while preserving aggregate rates |
| Fairness through awareness | Similar individuals should receive similar outcome distributions | Everything depends on whether the task-specific similarity metric is defensible |

The paper's proposal is normative, not merely predictive. A similarity metric is supposed to describe which people **ought** to be treated alike for a particular decision. It therefore cannot be justified only by saying that it predicts historical decisions well.

### 1.2 The motivating institutional split

The paper often separates two actors:

- A **data owner** possesses information about individuals and can enforce the fairness constraint.
- A **vendor** wants a useful classification and supplies, or helps define, a loss function.

This split lets the vendor optimize business goals without being trusted to decide what counts as fair. The fairness rule acts as a hard boundary on the vendor's freedom.

### Mental picture: a speed limit, not a destination

The fairness metric does not prescribe one classifier. It places a speed limit on how quickly treatment may change as we move between similar people.

Many different routes remain possible inside that limit. A lender may prefer one fair classifier, while a public-interest organization may prefer another, because they assign different losses to outcomes. The metric constrains the acceptable set; the utility function chooses within it.

### Check your understanding

1. Why can deleting a protected attribute fail even if the deletion is perfectly implemented?
2. What is the difference between learning who received loans historically and deciding who ought to be considered similarly creditworthy?

---

## 2. The mathematical objects

The framework has four main ingredients.

The formal setup and optimization problem are developed in §2 (pp. 4–7).

| Symbol | Meaning |
|---|---|
| $V$ | Set of individuals to be classified |
| $A$ | Set of possible outcomes or actions |
| $d:V\times V\to \mathbb{R}_{\ge 0}$ | Task-specific distance between individuals |
| $L:V\times A\to \mathbb{R}$ | Loss incurred when individual $x$ receives outcome $a$ |

The simplest nontrivial outcome set is $A=\{0,1\}$, but the formulation permits many outcomes: different advertisements, products, ratings, or policy choices.

### 2.1 A randomized classifier

The classifier is a mapping

$$
M:V\to \Delta(A),
$$

where $\Delta(A)$ is the set of probability distributions over outcomes.

For each individual $x$, write

$$
\mu_x=M(x).
$$

The value $\mu_x(a)$ is the probability that $x$ receives outcome $a$.

For a binary outcome, the whole distribution is determined by one number:

$$
p_x=\Pr[M(x)=1], \qquad \mu_x=(1-p_x,p_x).
$$

The framework therefore compares people through their **lotteries over outcomes**, not through a single random draw.

### 2.2 Why randomization matters

Suppose two nearly identical applicants sit on opposite sides of a deterministic threshold. One always receives a loan and the other never does. Their treatment distributions are maximally different even though their relevant characteristics are almost the same.

A randomized classifier can change approval probability smoothly. If one applicant has approval probability $0.72$, a very similar applicant might have probability $0.70$. A realized decision can still differ, but the policy does not systematically assign radically different chances to similar people.

This distinction is essential:

- **Ex post equality** asks whether two realized outcomes happened to match.
- **Ex ante fairness** asks whether two individuals faced appropriately similar distributions before randomness was resolved.

The paper formalizes the second idea.

### 2.3 The task-specific distance

The value $d(x,y)$ quantifies how different $x$ and $y$ are for the decision at hand. A small value means the task provides little justification for different treatment.

The same pair of people may have different distances in different tasks. Two people might be similar for receiving an advertisement for a general checking account but dissimilar for a specialized medical intervention.

The paper typically speaks of a metric, though a footnote observes that much of the setup only needs nonnegativity, symmetry, and $d(x,x)=0$. Later results, especially those using Earthmover distance and triangle inequalities, do use metric structure.

### 2.4 The loss function

The loss $L(x,a)$ says how undesirable it is, from the optimization perspective, to assign outcome $a$ to individual $x$.

The expected loss for one person is

$$
\mathbb{E}_{a\sim \mu_x}[L(x,a)]
=\sum_{a\in A}\mu_x(a)L(x,a).
$$

Averaging over the population gives

$$
\mathbb{E}_{x\sim V}\mathbb{E}_{a\sim\mu_x}[L(x,a)].
$$

The paper treats this objective abstractly. A lender's loss may reward profitable matches; another institution may encode different goals. The fairness constraint is meant to remain in force regardless of the chosen loss.

### Worked example: three loan applicants

Let $A=\{0,1\}$, where $1$ means showing a favorable loan offer. Suppose:

$$
d(x,y)=0.10,\qquad d(y,z)=0.20,\qquad d(x,z)=0.30.
$$

A candidate policy assigns

$$
p_x=0.75,\qquad p_y=0.70,\qquad p_z=0.50.
$$

Under total variation distance for binary outcomes, the distance between two outcome distributions is simply the absolute difference in positive-outcome probabilities. Therefore:

$$
|p_x-p_y|=0.05\le 0.10,
$$

$$
|p_y-p_z|=0.20\le 0.20,
$$

$$
|p_x-p_z|=0.25\le 0.30.
$$

Every pair satisfies the fairness constraint. If instead $p_z=0.20$, then $|p_y-p_z|=0.50>0.20$; the policy would treat $y$ and $z$ too differently relative to their task-specific distance.

### Check your understanding

1. Why does the framework compare probability distributions instead of only comparing final decisions?
2. If every person receives exactly the same lottery over outcomes, is the classifier fair under this definition? Is it necessarily useful?

---

## 3. Individual fairness as a Lipschitz condition

The key definition is compact:

$$
D(M(x),M(y))\le d(x,y) \qquad \text{for all }x,y\in V.
\tag{1}
$$

Here:

- $d(x,y)$ measures distance between people for the task.
- $D(M(x),M(y))$ measures distance between their outcome distributions.
- The inequality says that treatment is not allowed to vary faster than justified similarity permits.

A mapping satisfying Equation (1) is called $(D,d)$-**Lipschitz**.

### 3.1 Reading the inequality

Consider three cases.

1. If $d(x,y)=0$, then $D(M(x),M(y))=0$. The two people must receive identical outcome distributions.
2. If $d(x,y)$ is small, only a small treatment difference is allowed.
3. If $d(x,y)$ is large, the constraint becomes weak and the utility objective has more freedom.

The definition is local and pairwise. It does not begin with protected groups. It asks how every pair of individuals may be treated.

### 3.2 A geometric interpretation

Think of $V$ as a landscape. Nearby points are similar people. The classifier assigns a distribution to every point.

The Lipschitz rule prevents a cliff in treatment: walking a short distance through the individual space cannot produce a large jump in outcome distribution. A useful classifier may still slope, but it must slope gradually enough to respect $d$.

### 3.3 Two distances between outcome distributions

The paper studies two choices for $D$.

#### Total variation distance

For distributions $P$ and $Q$ over a finite set $A$,

$$
D_{\mathrm{tv}}(P,Q)
=\frac12\sum_{a\in A}|P(a)-Q(a)|.
\tag{2}
$$

Term by term:

- For each outcome $a$, compare the probability assigned by $P$ and $Q$.
- Take the absolute value so positive and negative differences do not cancel.
- Sum across outcomes.
- Divide by two because probability mass moved away from one outcome reappears elsewhere and would otherwise be counted twice.

Total variation has an operational meaning: it is the largest difference between the probabilities that $P$ and $Q$ assign to the same event.

For binary outcomes,

$$
D_{\mathrm{tv}}((1-p,p),(1-q,q))=|p-q|.
$$

**Worked calculation.** If $P=(0.7,0.3)$ and $Q=(0.5,0.5)$, then

$$
D_{\mathrm{tv}}(P,Q)
=\frac12(|0.7-0.5|+|0.3-0.5|)
=\frac12(0.2+0.2)=0.2.
$$

#### Relative $\ell_\infty$ distance

The paper also defines

$$
D_\infty(P,Q)
=\sup_{a\in A}
\log\max\left\{
\frac{P(a)}{Q(a)},
\frac{Q(a)}{P(a)}
\right\}.
\tag{3}
$$

This distance looks at the worst multiplicative change across outcomes.

For each outcome $a$:

1. Form the probability ratio in both directions.
2. Keep the larger ratio, so the comparison is symmetric.
3. Take the logarithm, turning multiplicative changes into additive distances.
4. Keep the worst outcome.

If $D_\infty(P,Q)\le \delta$, then for every outcome $a$,

$$
e^{-\delta}Q(a)\le P(a)\le e^\delta Q(a).
$$

This choice is sensitive to relative changes. Changing a probability from $0.001$ to $0.01$ is only an absolute change of $0.009$, but it is a tenfold multiplicative change.

### 3.4 Comparing the two choices

| Property | $D_{\mathrm{tv}}$ | $D_\infty$ |
|---|---|---|
| Measures | Absolute probability-mass difference | Worst multiplicative probability change |
| Natural scale | Between $0$ and $1$ | Between $0$ and $\infty$ |
| Binary simplification | $\lvert p-q\rvert$ | Worst log-ratio for the two outcomes |
| Zero-probability issue | Can remain finite | Can become infinite under support mismatch |
| Connection emphasized by paper | Earthmover characterization | Differential privacy |

The paper records the useful relation

$$
D_{\mathrm{tv}}(P,Q)
\le 1-e^{-D_\infty(P,Q)}
\le D_\infty(P,Q).
\tag{4}
$$

Therefore a $D_\infty$-based fairness guarantee also implies a total-variation guarantee at the same numerical distance.

### 3.5 Fairness survives post-processing

Suppose $M$ is fair and a later procedure $f$ transforms its outcomes, possibly using randomness. The composition $f\circ M$ remains Lipschitz for both distances studied in the paper.

The intuition is that processing two distributions through the same channel cannot make them easier to distinguish. If two individuals receive similar distributions at the protected interface, a vendor cannot create a larger distinction merely by relabeling or randomizing those outcomes afterward.

This guarantee has a boundary: post-processing must operate on the protected output. If the vendor also obtains outside information about the individual, the conclusion need not follow.

### Check your understanding

1. Why is the Lipschitz condition stronger than saying that protected-group membership is not an input?
2. Give a pair of distributions that are close in total variation but far in $D_\infty$.
3. What assumption makes the post-processing guarantee relevant in an institutional deployment?

---

## 4. The Fairness LP: optimize utility inside the fair region

The framework chooses the lowest-loss classifier subject to the Lipschitz constraints:

$$
\begin{aligned}
\operatorname{opt}(I)=
\min_{\{\mu_x\}_{x\in V}}
&\quad
\mathbb{E}_{x\sim V}
\mathbb{E}_{a\sim\mu_x}[L(x,a)]\\
\text{subject to}
&\quad D(\mu_x,\mu_y)\le d(x,y)
&&\forall x,y\in V,\\
&\quad \mu_x\in\Delta(A)
&&\forall x\in V.
\end{aligned}
\tag{5}
$$

### 4.1 What is being optimized?

There is one probability variable $\mu_x(a)$ for every individual–outcome pair.

The objective can be expanded as

$$
\frac{1}{|V|}
\sum_{x\in V}\sum_{a\in A}\mu_x(a)L(x,a)
$$

for a uniform population distribution.

The simplex constraints require

$$
\mu_x(a)\ge 0,
\qquad
\sum_{a\in A}\mu_x(a)=1.
$$

The pairwise constraints cut away policies that change treatment more sharply than the individual metric allows.

### 4.2 Why this becomes a linear program

With total variation, absolute values can be represented by auxiliary variables and linear inequalities. The objective is already linear in $\mu_x(a)$.

With $D_\infty$, the condition can be written outcome by outcome as

$$
\mu_x(a)\le e^{d(x,y)}\mu_y(a)
\qquad
\forall x,y,a,
\tag{6}
$$

together with the reverse-direction constraint. The constants $e^{d(x,y)}$ are fixed, so these inequalities are linear in the decision variables.

The paper therefore shows that, for finite $V$ and $A$, the optimum can be computed by a linear program of polynomial size.

### 4.3 Feasibility is easy; usefulness is not

A constant classifier that assigns every person the same distribution is always Lipschitz:

$$
M(x)=P \qquad \forall x\in V.
$$

Every pair has treatment distance zero. The fairness problem is therefore not whether some fair classifier exists. It is how much useful differentiation can remain after fairness is enforced.

### Worked example: fairness changes the utility optimum

Consider two similar applicants $x$ and $y$ with

$$
d(x,y)=0.1.
$$

Suppose the loss-minimizing unconstrained policy would assign

$$
p_x=1,\qquad p_y=0.
$$

Its total-variation treatment distance is $1$, which exceeds $0.1$. A fair binary policy must satisfy

$$
|p_x-p_y|\le 0.1.
$$

The optimizer may choose $(p_x,p_y)=(1,0.9)$, $(0.1,0)$, or something between, depending on the two loss functions. Fairness restricts the difference but does not decide which person's probability should move more.

This example reveals the division of labor:

- The metric determines the maximum justified gap.
- The loss function determines where the pair is placed inside that gap.

### 4.4 Computational and statistical caveats

Polynomial time does not automatically mean practical scale. The formulation can contain pairwise constraints for every $(x,y)$, which is quadratic in the number of individuals.

The paper also flags a generalization problem. The LP is written over the actual finite set $V$. If one only sees a sample, a further method is needed to extend a fair mapping to unseen people. The theoretical optimization formulation does not by itself solve metric learning, sampling, or out-of-sample generalization.

### Check your understanding

1. Why is a constant classifier always fair under the Lipschitz definition?
2. In the two-applicant example, what information would determine whether the optimizer moves $p_x$, $p_y$, or both?
3. Why does “polynomial-size LP” not settle the deployment problem?

---

## 5. Group fairness: what statistical parity says and misses

The paper's definition, counterexamples, and connection to individual fairness appear in §3 (pp. 7–11).

Let $S$ and $T$ be distributions over individuals, often representing a protected group and a comparison group. Average the individual output distributions within each group:

$$
\mu_S(a)=\mathbb{E}_{x\sim S}[\mu_x(a)],
\qquad
\mu_T(a)=\mathbb{E}_{x\sim T}[\mu_x(a)].
$$

The classifier satisfies **statistical parity up to bias $\varepsilon$** when

$$
D_{\mathrm{tv}}(\mu_S,\mu_T)\le \varepsilon.
\tag{7}
$$

If $\varepsilon=0$, group membership does not change the distribution of classified outcomes. For any event $O\subseteq A$, members of $S$ and $T$ have the same probability of receiving an outcome in $O$.

### 5.1 Why statistical parity is attractive

Statistical parity addresses an important aggregate question: do groups receive different outcome distributions?

It also blocks a simple redundant encoding at the output. If outcome distributions are identical by group, observing the outcome alone does not reveal group membership better than the prior distribution does.

### 5.2 Why parity alone is insufficient

The paper gives three mechanisms by which group totals can look right while individuals are treated badly.

| Failure mode | What parity gets right | What still goes wrong |
|---|---|---|
| Reduced utility | Equal aggregate selection rates | The classifier selects less-suitable members from one group because the same signal has different meaning across groups |
| Self-fulfilling prophecy | Enough protected-group members are selected | The decision maker deliberately selects poorly matched members, then uses their failure to justify later exclusion |
| Subset targeting | The protected group as a whole receives equal exposure | A harmful or strategically chosen subset within the group is targeted differently |

#### Example 1: the same credential can mean different things

Suppose cultural patterns steer the strongest members of group $S$ toward engineering and weaker members toward finance, while the pattern is reversed in $T$. A selector using “finance background” as a group-blind rule might meet parity yet choose the wrong subset of $S$.

The defect arises from ignoring group context, not from explicitly using group membership. Blindness and parity can coexist with low individual utility.

#### Example 2: selecting people expected to fail

An organization required to interview enough minority candidates could invite candidates it considers least competitive. The aggregate interview rate meets the target, while the internal selection undermines the purpose of the intervention.

#### Example 3: parity does not automatically descend to subgroups

An advertisement can be shown at equal rates to $S$ and $T$, while being shown to likely customers inside $S$ and unlikely customers inside $T$. Clicking then becomes highly correlated with group membership, even though exposure obeyed parity.

### 5.3 Individual and group fairness answer different questions

| Individual fairness | Statistical parity |
|---|---|
| Are relevantly similar people treated similarly? | Do groups receive similar aggregate outcomes? |
| Requires a similarity metric | Requires group definitions |
| Constrains pairwise treatment | Constrains group averages |
| Can preserve within-group merit structure | Can force an aggregate distribution |
| May fail to equalize groups far apart under the metric | May conceal unfair selection within a group |

The paper does not simply declare one side correct. It asks when the individual constraint already implies the group constraint and how to proceed when it does not.

### Check your understanding

1. Construct a policy with equal selection rates for two groups that is nevertheless unfair to individuals within one group.
2. Why can “use the same rule for everyone” produce reduced utility for one group?

---

## 6. Earthmover distance links individual and group fairness

The bridge between the two fairness notions is an optimal transportation problem.

Imagine the distribution $S$ as piles of earth located at individuals in the metric space. The target distribution $T$ specifies where that earth must end up. Moving one unit of mass from $x$ to $y$ costs $d(x,y)$.

A transport plan $h(x,y)$ records how much mass moves from $x$ to $y$. Its cost is

$$
\sum_{x,y\in V}h(x,y)d(x,y).
$$

The $d$-Earthmover distance is the cheapest feasible transport:

$$
d_{\mathrm{EM}}(S,T)
=
\min_h
\sum_{x,y\in V}h(x,y)d(x,y),
\tag{8}
$$

subject to the transport plan having source marginal $S$ and target marginal $T$:

$$
\sum_y h(x,y)=S(x),
\qquad
\sum_x h(x,y)=T(y),
\qquad
h(x,y)\ge 0.
$$

### 6.1 Why transport is the right object

Suppose a unit of probability mass from group $S$ at $x$ is paired with mass from group $T$ at $y$.

Individual fairness guarantees

$$
D_{\mathrm{tv}}(M(x),M(y))\le d(x,y).
$$

Average this inequality across all pairs in a coupling. A low-cost coupling pairs most of $S$'s mass with nearby mass from $T$. Nearby pairs must receive similar treatment, so the two groups' average output distributions must also be similar.

Earthmover distance therefore asks the exact structural question needed here: **Can the members of one group be matched, in distribution, to relevantly similar members of the other group?**

### 6.2 Worst-case group bias

For binary outcomes, the paper defines the maximum statistical-parity violation possible among all fair mappings:

$$
\operatorname{bias}_{D,d}(S,T)
=
\max_{M\text{ is }(D,d)\text{-Lipschitz}}
\left(\mu_S(0)-\mu_T(0)\right).
\tag{9}
$$

Restricting the definition to binary outputs loses no power for measuring total-variation group disparity. Given any larger outcome set, one can collect all outcomes more likely under $S$ into one super-outcome and the remainder into the other.

### 6.3 The central characterization

For total variation, the paper proves

$$
\operatorname{bias}_{D_{\mathrm{tv}},d}(S,T)
\le d_{\mathrm{EM}}(S,T).
\tag{10}
$$

If all distances are at most $1$, the bound is tight:

$$
\operatorname{bias}_{D_{\mathrm{tv}},d}(S,T)
=d_{\mathrm{EM}}(S,T).
\tag{11}
$$

So, in the bounded-metric setting, the maximum group disparity compatible with individual fairness is exactly the cost of optimally transporting one group's distribution to the other.

For $D_\infty$, the paper obtains the upper bound

$$
\operatorname{bias}_{D_\infty,d}(S,T)
\le d_{\mathrm{EM}}(S,T),
\tag{12}
$$

but leaves a tight characterization as an open question.

### 6.4 Worked transport example

Suppose $S$ is uniform on $s_1,s_2$ and $T$ is uniform on $t_1,t_2$. The cross-group distances are:

|  | $t_1$ | $t_2$ |
|---|---:|---:|
| $s_1$ | 0.1 | 0.9 |
| $s_2$ | 0.8 | 0.2 |

The cheapest plan sends the $0.5$ mass at $s_1$ to $t_1$ and the $0.5$ mass at $s_2$ to $t_2$. Its cost is

$$
d_{\mathrm{EM}}(S,T)
=0.5(0.1)+0.5(0.2)=0.15.
$$

Every total-variation-Lipschitz classifier must therefore have statistical-parity bias at most $0.15$ between these groups. When distances are bounded by $1$, some fair binary mapping attains this worst-case value, so the guarantee cannot be improved without extra assumptions.

If the cheap diagonal matches did not exist and all cross-group distances were near $1$, individual fairness would say little about group parity. The framework protects pairs declared similar; it cannot create cross-group similarity that the metric denies.

### 6.5 What the theorem does—and does not—say

The theorem does **not** say individual fairness always implies statistical parity. It gives a condition:

$$
\text{small Earthmover distance between groups}
\Longrightarrow
\text{small parity gap for every fair classifier}.
$$

Conversely, under bounded total-variation geometry, a large Earthmover distance permits a large parity gap.

The result moves political weight onto the metric. If a metric systematically places members of two social groups far apart, the individual-fairness constraint will permit different group outcomes. Whether that geometry is justified is not answered by the optimization theorem.

### Check your understanding

1. Why is average pairwise distance between random members of $S$ and $T$ not the same as Earthmover distance?
2. If $d_{\mathrm{EM}}(S,T)=0$, what follows about every Lipschitz classifier's group outcomes?
3. What normative concern arises if historical inequality is encoded into the metric and creates a large Earthmover distance?

---

## 7. Fair affirmative action

When two groups are far apart under the chosen metric, Lipschitz fairness may not imply parity. Simply adding parity as another hard constraint can also collapse the solution into nearly identical treatment for everyone, sacrificing useful distinctions.

The construction summarized here appears in §4 (pp. 11–15).

The paper studies this tension through a stylized example.

### 7.1 The two-cluster example

Let $S$ be 10% of the population and $T$ the remaining 90%. Suppose the metric divides people into two tight clusters, $G_0$ and $G_1$, with large distance between clusters.

Most of $S$ lies in $G_0$, while most of $T$ lies in $G_1$. Imagine $G_1$ is offered a better loan advertisement than $G_0$.

Within-cluster Lipschitz fairness is easy: people close to one another receive similar treatment. But group outcomes differ because the groups occupy different parts of the metric space.

If exact parity is imposed while every cross-group Lipschitz constraint remains, the links can force both clusters toward the same treatment distribution. The space effectively collapses, leaving a fair but low-utility constant policy.

If parity is not imposed, a hostile vendor can exploit the separation. It might replace the less attractive offer shown in $G_0$ with a message designed to drive customers away, thereby excluding much of $S$ while preserving treatment similarity inside the cluster.

### 7.2 The paper's two-stage construction

The proposed compromise preserves Lipschitz fairness **within** each group, enforces statistical parity, and treats cross-group Lipschitz constraints as a quantity to minimize rather than a hard requirement.

Let $S$ be the protected group and $T$ the comparison group.

#### Stage 1: transport each member of $S$ into a distribution over $T$

For every $x\in S$, find a distribution $\mu_x\in\Delta(T)$. The collection should:

1. Send the uniform distribution on $S$ close to the uniform distribution on $T$.
2. Preserve Lipschitz similarity among members of $S$.
3. Minimize expected transport distance.

Conceptually, each protected-group member is represented by a lottery over similar comparison-group members.

#### Stage 2: solve a fairness problem on $T$

The loss for $y\in T$ is reweighted to include the losses of protected-group members transported to $y$:

$$
L'(y,a)
=L(y,a)+\sum_{x\in S}\mu_x(y)L(x,a).
\tag{13}
$$

Then solve the original Fairness LP on $T$ using $L'$. Let $\nu_y$ denote the resulting distribution over outcomes for $y\in T$.

#### Compose the maps

Define the final classifier by

$$
M(x)=
\begin{cases}
\nu_x, & x\in T,\\[4pt]
\mathbb{E}_{y\sim\mu_x}[\nu_y], & x\in S.
\end{cases}
\tag{14}
$$

A member of $T$ receives the outcome distribution computed directly for that person. A member of $S$ receives a mixture of the distributions assigned to the comparison-group representatives in that person's transport lottery.

### 7.3 What the construction guarantees

The paper proves that the composed mapping:

1. Satisfies statistical parity between $S$ and $T$, up to the chosen tolerance $\varepsilon$.
2. Preserves the Lipschitz condition for pairs within $S$ and for pairs within $T$.
3. Bounds the average cross-group Lipschitz violation by the cost of the restricted transport problem.

The design deliberately gives up a universal pairwise guarantee across $S\times T$. Cross-group distortion is controlled through the optimization objective instead.

### 7.4 Why this is not just “add a quota”

Adding parity while deleting all cross-group constraints allows the vendor to choose a strategically bad subset of $S$. The transport step uses the metric to preserve some relationship between individuals across groups. It therefore aims to equalize group outcomes without abandoning all individual structure.

The proposal is a bicriteria optimization:

- minimize decision loss;
- minimize disruption of cross-group similarity;
- subject to parity and within-group Lipschitz fairness.

Different applications can weight those objectives differently.

### 7.5 An alternative: repair the metric

The paper also sketches a different approach. Historical disadvantage may mean the original metric underestimates potential. Instead of changing classification after accepting the metric, construct a new metric $d'$ that stays close to $d$ while making the groups' Earthmover distance small.

This view treats affirmative action as a correction to the comparison standard itself. It asks whether a strong member of $S$ should be compared to a strong member of $T$, even when historically observed features place them farther apart.

The paper leaves open how to define “best approximates” and how to choose the appropriate correction. Those are substantive normative and domain questions, not mere numerical tuning.

### Worked micro-example

Suppose $S=\{s_1,s_2\}$ and $T=\{t_1,t_2\}$. Stage 1 chooses

$$
\mu_{s_1}(t_1)=0.8,\quad \mu_{s_1}(t_2)=0.2,
$$

$$
\mu_{s_2}(t_1)=0.2,\quad \mu_{s_2}(t_2)=0.8.
$$

Averaging the two lotteries gives $(0.5,0.5)$, the uniform distribution on $T$, so aggregate transport supports parity.

If Stage 2 assigns positive-outcome probabilities

$$
p_{t_1}=0.9,\qquad p_{t_2}=0.4,
$$

then the composed probabilities are

$$
p_{s_1}=0.8(0.9)+0.2(0.4)=0.80,
$$

$$
p_{s_2}=0.2(0.9)+0.8(0.4)=0.50.
$$

The group average for $S$ is $0.65$, exactly matching the average $(0.9+0.4)/2=0.65$ for $T$. Yet the two members of $S$ are not treated identically; the transport preserves individual differentiation.

### Check your understanding

1. Why can imposing parity together with every original Lipschitz constraint force a nearly constant classifier?
2. Which fairness guarantee is deliberately relaxed in the two-stage construction?
3. How does the transport step reduce the risk of choosing the “wrong” subset of the protected group?

---

## 8. The connection to differential privacy

The paper observes that its fairness definition generalizes the mathematical form of differential privacy.

The basic analogy is introduced in §2.3 (pp. 6–7), and the exponential-mechanism analysis appears in §5 (pp. 15–18).

### 8.1 The shared pattern

| Fairness setting | Differential-privacy setting |
|---|---|
| Input is an individual $x$ | Input is a database $x$ |
| Nearby inputs are similar people | Nearby inputs are databases differing in few records |
| Output is a classification distribution | Output is a randomized answer distribution |
| Nearby individuals should see similar treatment | Neighboring databases should induce similar observable outputs |
| Metric expresses legitimate task difference | Metric scales with number of changed records and privacy parameter |

Let databases be subsets of a universe $U$, so $V=2^U$. Let $x\triangle y$ be their symmetric difference. Define

$$
d(x,y)=\varepsilon |x\triangle y|.
\tag{15}
$$

Then $\varepsilon$-differential privacy is precisely a $(D_\infty,d)$-Lipschitz condition in this metric representation.

For neighboring databases differing in one person's record,

$$
d(x,y)=\varepsilon,
$$

so every output probability may change by at most a multiplicative factor $e^\varepsilon$.

### 8.2 The goals remain different

The mathematical analogy should not erase the conceptual difference.

- Differential privacy aims to limit what an observer can infer about the contribution of a person's record.
- Individual fairness aims to limit how much a decision rule may vary between people deemed similar for a task.

Privacy defines neighbors structurally through databases. Fairness needs a socially and institutionally defensible similarity metric over people.

### 8.3 The exponential mechanism

Borrowing from differential privacy, the paper considers a mechanism mapping each individual to a distribution over $V$ itself:

$$
E(x)(y)=\frac{e^{-d(x,y)}}{Z_x},
\qquad
Z_x=\sum_{z\in V}e^{-d(x,z)}.
\tag{16}
$$

The mechanism favors outputs near $x$:

- An output at distance $0$ receives unnormalized weight $1$.
- An output at distance $1$ receives weight $e^{-1}$.
- An output at distance $2$ receives weight $e^{-2}$.

The normalizer $Z_x$ turns these weights into probabilities. The paper states that this exponential mechanism is $(D_\infty,d)$-Lipschitz.

### Worked example: exponential decay

Suppose $x$ has three possible representative outputs at distances $0$, $1$, and $2$. Their unnormalized weights are

$$
1,\quad e^{-1}\approx 0.368,\quad e^{-2}\approx 0.135.
$$

The normalizer is approximately

$$
Z_x=1+0.368+0.135=1.503.
$$

The output probabilities are approximately

$$
0.665,\quad 0.245,\quad 0.090.
$$

The mechanism usually selects a nearby output but does not deterministically reveal the exact input. This smoothness is what makes the same construction useful in both privacy and fairness analyses.

### Check your understanding

1. What plays the role of “neighboring inputs” in fairness and in differential privacy?
2. Why does the shared Lipschitz form not imply that fairness and privacy are the same social objective?

---

## 9. Utility bounds and doubling dimension

The general Fairness LP finds the best fair classifier for a finite instance, but it does not give a simple quantitative bound on loss. The paper obtains such bounds for the exponential mechanism when the metric space has controlled geometric growth.

### 9.1 Doubling dimension

A metric space has doubling dimension $k$ if every ball of radius $R$ can be covered by at most $2^k$ balls of radius $R/2$.

Formally, with

$$
B(x,R)=\{y\in V:d(x,y)\le R\},
$$

the smallest such $k$ is the doubling dimension.

### Mental picture

On a line, an interval of radius $R$ can be covered by a small constant number of intervals of radius $R/2$. In a high-dimensional space, many more half-radius balls may be needed.

Doubling dimension therefore captures intrinsic geometric complexity. Low dimension means neighborhoods do not explode in size too rapidly as radius grows.

### 9.2 Well-separated spaces

The paper calls a finite metric space well separated if there is some $\eta>0$ such that every radius-$\eta$ ball contains only its center:

$$
|B(x,\eta)|=1 \qquad \forall x\in V.
$$

This rules out arbitrarily dense clusters of distinct points at vanishing distance. If a space is not well separated, the paper notes that one can work with a separated subset and map each excluded point to a nearby representative, paying a small additive loss and a small additive fairness relaxation.

### 9.3 Upper bound

When $(V,d)$ is well separated and has bounded doubling dimension, the paper proves that the exponential mechanism has constant average loss:

$$
\mathbb{E}_{x\sim V}
\mathbb{E}_{y\sim E(x)}[d(x,y)]
=O(1).
\tag{17}
$$

The proof's intuition is a competition between two forces:

- The number of possible outputs grows as one looks farther from $x$.
- Each output's weight shrinks exponentially as $e^{-d(x,y)}$.

In a bounded-doubling space, neighborhood growth is controlled strongly enough that exponential decay wins. Faraway outcomes contribute little total expected distance.

The bound's hidden dependence on dimension is not gentle: the proof gives exponential dependence on the doubling dimension. “$O(1)$” here assumes dimension is treated as a constant.

### 9.4 Lower bound

The paper also shows that dependence on dimension cannot disappear entirely. For every $k\ge 2$ and sufficiently large $n$, there exists an $n$-point metric space with doubling dimension $O(k)$ such that every $(D_\infty,d)$-Lipschitz mapping has average loss at least

$$
\Omega(k).
\tag{18}
$$

The proof uses a packing construction on a high-dimensional sphere. Many well-separated regions compete for probability mass. Lipschitz fairness forces a mechanism that is accurate around many inputs to allocate too much total probability, producing a contradiction. Some average error proportional to dimension is unavoidable.

### 9.5 The quantitative tradeoff

The two theorems bracket the problem:

| Result | Message |
|---|---|
| Upper bound for exponential mechanism | Low-complexity metric spaces permit an explicit fair mechanism with controlled loss |
| Lower bound for any Lipschitz mechanism | Geometric complexity can force utility loss; no clever mechanism eliminates it completely |

This is a formal fairness–utility tradeoff. It does not say fairness always imposes a fixed cost. It says the cost depends on the geometry induced by the task-specific metric.

### Check your understanding

1. Why does exponential probability decay help only if the number of distant candidates does not grow too quickly?
2. What does the lower bound rule out?
3. Why should a practitioner be cautious when reading the upper bound as “constant loss”?

---

## 10. The metric is the framework's moral center

The optimization is clean once $d$ is supplied. Supplying $d$ is the hard part.

The paper discusses the metric throughout §1 and returns to construction questions in §6.1 (pp. 18–20).

The paper says the metric should ideally represent ground truth for the task. When ground truth is unavailable, it imagines a publicly debated best approximation, perhaps imposed by regulators or proposed by civil-rights organizations. It advocates making metrics visible rather than leaving them hidden inside institutional practice.

### 10.1 Why a learned metric is not automatically legitimate

A metric learned from historical decisions may reproduce the judgments that fairness was meant to constrain. For example, learning applicant similarity from past lending decisions can encode redlining rather than creditworthiness.

A prediction metric answers:

> Which people had similar recorded outcomes in the available data?

A fairness metric must answer:

> Which people deserve similar treatment for this decision?

The first question can inform the second, but cannot settle it.

### 10.2 Within-group and cross-group comparisons

The paper suggests that machine-learning methods may help infer distances among members of the same group. Cross-group comparisons may need more human and domain input because features and opportunities can have different meanings across social contexts.

This is not merely a data-sparsity issue. A score gap may reflect differences in opportunity, measurement quality, or past discrimination. Deciding whether it represents a legitimate task difference requires causal and normative judgment.

### 10.3 Metric labeling

One proposed construction begins with a metric on the comparison population and then maps protected-group members into it using expert or domain information. This resembles the metric-labeling problem: known metric points serve as labels, and new objects must be assigned to them while respecting observed constraints.

The paper also asks how many expert queries are needed to approximate an unknown correct metric $d^*$ within bounded multiplicative distortion. This connects fairness-metric construction to spanners and metric embeddings.

The question remains open-ended because approximate recovery is only valuable if $d^*$ itself has a defensible meaning.

### 10.4 User participation

The paper briefly imagines letting users specify attributes they do or do not want considered. A menu of metrics might protect different attributes.

This idea faces a redundancy problem. A user may exclude one variable while other variables encode it. It also creates a consistency problem: if each person uses a different metric, it becomes unclear how to define pairwise fairness between them.

### 10.5 A metric audit

Before treating a fairness metric as valid, ask:

- What task is the metric specific to?
- Which attributes contribute to distance, and why are they relevant?
- Were distances learned from outcomes shaped by discrimination?
- Does measurement error differ across groups?
- Do within-group and cross-group distances have the same evidentiary basis?
- Who proposed, approved, and can contest the metric?
- How will the metric change as institutions and social conditions change?
- Which people become isolated in the geometry, with few nearby comparison points?
- Does the metric measure present performance, potential, need, desert, risk, or something else?

### Check your understanding

1. Why is publishing the metric valuable even if the metric is imperfect?
2. Give an example in which an accurate predictive metric would be an inappropriate fairness metric.
3. Who should participate in defining a task-specific metric for university admissions, and what kinds of evidence should they contribute?

---

## 11. What the framework prevents—and where its guarantees stop

The paper's appendix lists discriminatory practices that the framework seeks to block. The common thread is that a vendor should not be able to create a large treatment distinction where the metric says little relevant difference exists.

The catalog appears in Appendix A (pp. 22–23); the limits of parity as information hiding are discussed in §6.3 (pp. 20–21).

| Practice | Mechanism | How a good similarity metric helps |
|---|---|---|
| Explicit discrimination | Directly condition worse treatment on protected status | Cross-group similar pairs must receive similar distributions |
| Redundant encoding | Use proxies that reconstruct protected status | Proxy-based jumps still violate pairwise similarity constraints |
| Redlining | Use geography or another correlated feature to deny opportunity | Geographically encoded group status cannot justify treatment if it is irrelevant to $d$ |
| Reverse redlining | Target a protected or vulnerable group with disadvantageous products | Similarity constrains group-specific targeting when comparable people exist |
| Selective market exit | Stop serving a segment containing many protected people | Large outcome changes within a similarity neighborhood are forbidden |
| Wrong-subset selection | Satisfy a group total using strategically chosen members | Pairwise constraints preserve relevant within-group structure |
| Reverse tokenism | Treat non-protected people as interchangeable tokens to manipulate parity | Every person's similarity relationships remain binding |

These protections are conditional. If the metric treats the affected people as far apart, the classifier receives freedom to differentiate them. The guarantee is therefore only as strong as the metric's conception of relevance.

### 11.1 Statistical parity is not privacy

The paper also asks whether parity hides sensitive membership from an advertiser. Equal output distributions for a broad group can prevent direct inference from that output, but the property is not hereditary to all subgroups.

Protecting the category “HIV-positive,” for example, need not protect the narrower category “people with AIDS.” A utility-maximizing classifier might concentrate errors among HIV-positive people without AIDS while still allowing the narrower condition to be targeted.

Thus parity for one set does not automatically protect every correlated or nested sensitive set. Fair classification and information privacy require distinct analyses, even when they share mathematical tools.

### 11.2 Certification versus construction

The metric can also audit an existing classifier. Given $M$, examine pairwise violations:

$$
\max_{x,y\in V}
\left[D(M(x),M(y))-d(x,y)\right]_+.
$$

A positive value identifies treatment changing faster than the metric permits. This can reveal unfairness even when the classifier itself did not have access to the protected information used by auditors.

The audit still needs representative data and a legitimate metric, but it shows the framework is not limited to designing a classifier from scratch.

### Check your understanding

1. Why does the framework address proxy discrimination more directly than feature deletion does?
2. Why does parity for a broad protected category not guarantee privacy for its subgroups?

---

## 12. Contributions, limitations, and legacy

### 12.1 Main contributions

The paper contributes a coherent mathematical program:

1. Define individual fairness through a task-specific metric and a Lipschitz mapping into outcome distributions.
2. Optimize utility subject to that fairness rule using a linear program.
3. Show when pairwise fairness implies group statistical parity through Earthmover distance.
4. Explain, with concrete counterexamples, why parity alone is insufficient.
5. Give a transport-based construction that enforces parity while preserving within-group individual fairness.
6. Connect the fairness constraint to differential privacy and reuse the exponential mechanism.
7. Prove geometry-dependent upper and lower bounds on fair classification loss.

### 12.2 Limitations to remember

| Limitation | Why it matters |
|---|---|
| The metric is assumed rather than solved | Defining legitimate similarity is the central social and technical problem |
| Pairwise constraints can be numerous | Finite polynomial solvability may still be expensive at scale |
| The basic LP is transductive | It does not automatically extend to unseen individuals |
| Randomized fairness is ex ante | Two similar people may still receive different realized outcomes |
| Utility is represented by a supplied loss | A harmful institutional objective can remain harmful even when pursued fairly within the metric |
| Parity is defined for chosen groups | Nested, intersecting, or unanticipated subgroups may remain unprotected |
| Static formulation | Long-run feedback, strategic response, and changing social conditions are not modeled |
| Fair treatment is not full justice | Equal handling by a classifier does not repair unequal resources, opportunity, or institutional legitimacy |

### 12.3 Conceptual legacy

The paper established a durable vocabulary for **individual fairness**: treat similar individuals similarly. It also made the tension between individual and group fairness mathematically explicit instead of treating “fairness” as a single undifferentiated property.

Its most important lasting lesson is two-sided:

- A group statistic is too coarse to protect every individual.
- An individual rule is only as defensible as the similarity standard it enforces.

Later fairness research developed many additional criteria, learned metrics, causal approaches, subgroup guarantees, and impossibility results. Those developments do not remove the paper's question. They sharpen it: what differences should a decision system be allowed to act on, and who gets to decide?

---

## 13. Synthesis: the whole framework in one flow

```mermaid
flowchart TD
    A[Public or institutionally governed<br/>task-specific metric d]
    B[Vendor or decision-maker<br/>loss function L]
    C[Optimize mapping M: V to distributions over A]
    D[Hard pairwise constraint<br/>D(Mx, My) <= d(x,y)]
    E[Utility-optimal fair classifier]
    F{Are group distributions<br/>close in Earthmover distance?}
    G[Individual fairness already<br/>bounds statistical-parity gap]
    H[Consider fair affirmative action:<br/>enforce parity, preserve within-group<br/>Lipschitz fairness, minimize cross-group distortion]
    A --> C
    B --> C
    D --> C
    C --> E
    E --> F
    F -->|Yes| G
    F -->|No, but parity is required| H
```

The framework separates three questions that are often mixed together:

1. **Normative geometry:** who should count as similar for this task?
2. **Constrained optimization:** among policies respecting that geometry, which best serves the stated objective?
3. **Population consequences:** what group-level disparities remain, and should additional intervention alter the metric or relax some pairwise constraints?

The equations make the tradeoffs inspectable. They do not make the normative choices disappear.

---

## Glossary

**Bias between groups:** In this paper's formal analysis, the maximum difference in group outcome probabilities permitted among Lipschitz mappings.

**Coupling:** A joint distribution over pairs whose two marginals are specified distributions; in optimal transport, it describes how probability mass is matched.

**Data owner:** A trusted party that holds individual data and can enforce the fairness mapping before exposing outcomes to a vendor.

**Differential privacy:** A guarantee that nearby databases induce multiplicatively similar output distributions.

**Doubling dimension:** A measure of metric-space growth: every radius-$R$ ball can be covered by at most $2^k$ radius-$R/2$ balls.

**Earthmover distance:** The minimum cost of transporting one probability distribution into another when moving mass from $x$ to $y$ costs $d(x,y)$.

**Ex ante fairness:** Fairness of the probability distribution offered before a randomized outcome is realized.

**Exponential mechanism:** A randomized mapping that assigns output $y$ probability proportional to $e^{-d(x,y)}$.

**Fairness through awareness:** The strategy of explicitly representing task-relevant similarity and constraining treatment differences by it.

**Fairness through blindness:** The strategy of omitting protected attributes, which can fail when other features redundantly encode them.

**Individual fairness:** The principle that relevantly similar individuals should receive similar treatment.

**Lipschitz mapping:** A mapping whose output distance is no greater than its input distance: $D(M(x),M(y))\le d(x,y)$.

**Loss function:** A numerical representation of the cost of assigning a particular outcome to a particular individual.

**Metric:** A function describing distances among objects, ordinarily satisfying nonnegativity, identity, symmetry, and the triangle inequality.

**Post-processing:** Applying the same downstream transformation to a mechanism's outputs; the distances used in the paper do not increase under such processing.

**Protected group:** A population whose members are at risk of discrimination and for whom additional fairness constraints may be relevant.

**Randomized classifier:** A classifier that maps each individual to a probability distribution over outcomes rather than to one fixed outcome.

**Relative $\ell_\infty$ distance:** The worst symmetric log probability ratio across outcomes, denoted $D_\infty$.

**Statistical parity:** Similarity or equality of aggregate outcome distributions across groups.

**Task-specific similarity metric:** A normative representation of which differences among individuals are relevant to a particular classification task.

**Total variation distance:** Half the $\ell_1$ difference between two probability distributions; equivalently, the largest probability difference they assign to an event.

**Vendor:** The party that wants classifications and whose preferences are represented through the loss function.

---

## Review and exam-style questions

### Short answer

1. Explain “fairness through awareness” without using the words *metric* or *Lipschitz*.
2. Why is fairness through blindness vulnerable to redundant encodings?
3. Why does the paper use randomized classifiers?
4. What does $d(x,y)=0$ require of a fair classifier?
5. Compare total variation and $D_\infty$ in terms of absolute and relative probability changes.
6. Why is a constant classifier fair but potentially useless?
7. State one way statistical parity can coexist with unfair treatment of individuals.
8. What does Earthmover distance measure in the fairness setting?
9. When does total-variation individual fairness imply a small statistical-parity gap?
10. Which pairwise guarantees are preserved and which are relaxed in the paper's affirmative-action construction?
11. In what mathematical sense does differential privacy become a special case of the fairness framework?
12. What role does doubling dimension play in the exponential mechanism's utility?
13. Why is the similarity metric a normative object rather than just a learned representation?
14. Distinguish constructing a fair classifier from auditing an existing one.

### Analytical problems

1. **Binary Lipschitz audit.** Three individuals have positive-outcome probabilities $(0.8,0.6,0.3)$. Their pairwise distances are $d_{12}=0.15$, $d_{23}=0.35$, and $d_{13}=0.50$. Under total variation, identify every violated fairness constraint and compute the amount of each violation.

2. **Total variation.** Compute $D_{\mathrm{tv}}(P,Q)$ for $P=(0.5,0.3,0.2)$ and $Q=(0.4,0.1,0.5)$. Find an event $O$ whose probability difference achieves this distance.

3. **Multiplicative sensitivity.** Compare changing an event probability from $0.40$ to $0.50$ with changing it from $0.01$ to $0.11$. Both are absolute changes of $0.10$. Explain why $D_\infty$ reacts differently.

4. **Fairness LP.** For two individuals and two outcomes, write all simplex and total-variation Lipschitz constraints explicitly. Choose a loss function and solve the small program by inspection.

5. **Parity counterexample.** Construct two groups of four people each and a classifier with equal 50% selection rates. Make the classifier select the least qualified two members of one group and the most qualified two of the other. Explain which criterion is satisfied and which principle is violated.

6. **Earthmover calculation.** Two group distributions each place mass $1/2$ on two points. Use the cross-distance matrix

   $$
   \begin{pmatrix}
   0.2 & 0.7\\
   0.6 & 0.3
   \end{pmatrix}
   $$

   to find the minimum transport cost. What upper bound does the paper then give on total-variation group bias?

7. **Non-obvious transport.** Construct a $3\times 3$ distance matrix in which matching each source to its individually nearest target is not globally feasible or optimal because multiple sources compete for limited target mass.

8. **Affirmative-action composition.** Recompute the micro-example in Section 7 if $p_{t_1}=0.7$ and $p_{t_2}=0.1$. Verify parity and compare within-group differences.

9. **Exponential mechanism.** Normalize weights for outputs at distances $0,0.5,1.5,3$. Compute the expected distance and explain the effect of adding ten new outputs at distance $3$.

10. **Geometry and utility.** Explain why a very high-dimensional metric space can make the total probability of faraway outputs substantial even when each faraway output receives exponentially small weight.

### Discussion prompts

1. Who should have authority to define a similarity metric for lending, admissions, hiring, or medical care?
2. Is randomization ethically acceptable in high-stakes decisions? Compare it with deterministic thresholds that create discontinuities.
3. Can two people be treated “similarly” when they receive different realized outcomes but equal probabilities?
4. When should group parity override a task-specific similarity metric?
5. If a metric predicts repayment accurately but reflects unequal access to wealth, should it be used for fairness?
6. Does a transparent but controversial metric improve accountability compared with an opaque learned classifier?
7. How should intersectional groups change the paper's analysis of statistical parity?
8. What forms of long-run feedback are absent from the one-shot framework?
9. Could a vendor choose a formally acceptable loss function that produces socially harmful outcomes despite the fairness constraint?
10. Should fairness auditing use the same data available to the classifier, or may it use protected information unavailable at decision time?

---

## Further reading

- Cynthia Dwork et al., [“Fairness Through Awareness”](https://arxiv.org/abs/1104.3913): abstract, versions, and bibliographic record for the primary paper.
- Moritz Hardt, Eric Price, and Nati Srebro, [“Equality of Opportunity in Supervised Learning”](https://arxiv.org/abs/1610.02413): a prominent group-conditional criterion based on error rates.
- Jon Kleinberg, Sendhil Mullainathan, and Manish Raghavan, [“Inherent Trade-Offs in the Fair Determination of Risk Scores”](https://arxiv.org/abs/1609.05807): incompatibilities among group-level fairness properties.
- Sorelle A. Friedler, Carlos Scheidegger, and Suresh Venkatasubramanian, [“On the (Im)possibility of Fairness”](https://arxiv.org/abs/1609.07236): how assumptions connecting observed and construct spaces shape fairness.
- Solon Barocas, Moritz Hardt, and Arvind Narayanan, [*Fairness and Machine Learning*](https://fairmlbook.org/): a broader treatment of fairness criteria, measurement, and sociotechnical context.
- Cynthia Dwork and Aaron Roth, [*The Algorithmic Foundations of Differential Privacy*](https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf): background for the privacy concepts and mechanisms reused in the paper.

## One-sentence takeaway

**Fairness Through Awareness turns “treat similar people similarly” into an optimizable mathematical constraint, while making clear that the hardest question is who gets to define similarity.**
