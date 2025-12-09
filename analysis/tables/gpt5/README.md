# Hypothesis Testing Tables: GPT-5-mini

## Table Descriptions

### Overall Performance and Delta Metrics (H1)
**Hypothesis:** H1: Causal Graph Non-Utilization

**Notes:**
- Δ₁ (Graph Benefit) = Acc(Correct) - Acc(No Graph)
- Δ₂ (Corruption Harm) = Acc(Correct) - Acc(Corrupted)
- Δ₃ (Parametric Advantage) = Acc(No Graph) - Acc(Corrupted)
- All confidence intervals computed via bootstrap (10,000 iterations)
- Significance: CI excludes 0 → p < 0.05

### Corruption-Specific Sensitivity (H2)
**Hypothesis:** H2: Differential Corruption Sensitivity

**Notes:**
- Δ₂ = Acc(Correct Graph) - Acc(This Corruption)
- Sorted by Δ₂ magnitude (most harmful first)
- Expected ranking: Confounder > Collider > Mediator > Random Edge
- Positive Δ₂ indicates corruption hurts performance

### Scenario × Corruption Interaction (H3)
**Hypothesis:** H3: Scenario-Type Interaction

**Notes:**
- Δ₂ = Acc(Correct Graph) - Acc(Corrupted Graph) for each scenario
- Expected degradation: Nonsense > Anti-commonsense > Commonsense
- Parametric memory should provide more compensation for commonsense scenarios

### Rung-Level Analysis (H6)
**Hypothesis:** H6: Causal Reasoning Level Effects

**Notes:**
- Rung 1 (Association): P(Y|X) - observational queries
- Rung 2 (Intervention): P(Y|do(X)) - action-based queries
- Rung 3 (Counterfactual): P(Y_x|X',Y') - what-if queries
- Expected: Stronger graph effects at lower rungs (simpler reasoning)

### Error Pattern Analysis (H4, H8)
**Hypothesis:** H4: Corruption-Type Specificity, H8: Error Consistency

**Notes:**
- Yes→No: Model incorrectly predicted No when answer was Yes (false negative)
- No→Yes: Model incorrectly predicted Yes when answer was No (false positive)
- Ratio: (Yes→No) / (No→Yes) indicates bias direction
- Look for systematic patterns (e.g., colliders causing specific error types)

### Question-Level Correctness Patterns
**Hypothesis:** Supporting analysis for H1

**Notes:**
- Pattern format: [no_graph][original][corrupted] where 1=correct, 0=wrong
- Strict criterion: corrupted=1 only if ALL corruption types answered correctly

### Pattern Summary Metrics
**Hypothesis:** H1 supporting metrics

**Notes:**
- Net positive effect indicates graphs are beneficial overall
- Net negative effect indicates graphs are harmful overall

### Statistical Tests Summary
**Hypothesis:** Supplementary statistical tests

**Notes:**
- Test 1: Rung-level ANOVA tests if Δ₁ (graph benefit) varies significantly by rung
- Test 2: Rung-level ANOVA tests if Δ₂ (corruption harm) varies significantly by rung
- Test 3: All pairwise corruption type comparisons with Bonferroni correction
- Test 4: McNemar's test for paired binary outcomes (three comparisons)
- Test 5: Cohen's h effect sizes for Δ₁ (original vs. no_graph), Δ₂ (original vs. corrupted), and Δ₃ (no_graph vs. corrupted)
- Bootstrap iterations: 10000

## Data Summary

- Total valid responses: 7473
- Unique questions: 1246
- Conditions tested: 6
