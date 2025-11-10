# Part 3: Ethical Reflection

## Prompt
Your predictive model from Task 3 is deployed in a company. Discuss:

- Potential biases in the dataset (e.g., underrepresented teams).
- How fairness tools like IBM AI Fairness 360 could address these biases.

---

## Analysis

Predictive models can unintentionally inherit biases from the training data. For example:

- Certain teams or user groups may be underrepresented.
- Historical operational decisions may favor some departments over others.
- Imbalanced datasets may cause the model to consistently misclassify or deprioritize issues for underrepresented groups.

### Addressing Bias

Fairness tools like **IBM AI Fairness 360** can:

1. **Detect biases**: Measure disparities across groups using metrics like statistical parity, disparate impact, or equal opportunity.  
2. **Mitigate biases**: Apply techniques such as reweighing, adversarial debiasing, or post-processing adjustments to correct predictions.  
3. **Continuous monitoring**: Evaluate model predictions over time to ensure fairness is maintained.  

Using these tools, the company can deploy the predictive model responsibly, ensuring that all teams receive equitable treatment and resource allocation. Human oversight remains critical to validate decisions and maintain ethical standards.
