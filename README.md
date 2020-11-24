# QOSF-Mentorship-Task2
This repo holds my solution to Task 2 of the Screening Tasks for the Quantum Open Source Foundation Mentorship Program's second cohort.
Please use this link to view it in NBViewer:
https://nbviewer.jupyter.org/github/Jwala-1908/QOSF-Mentorship-Task2/blob/master/Task2.ipynb

Qiskit Version info
{'qiskit-terra': '0.15.2',
 'qiskit-aer': '0.6.1',
 'qiskit-ignis': '0.4.0',
 'qiskit-ibmq-provider': '0.9.0',
 'qiskit-aqua': '0.7.5',
 'qiskit': '0.19.6'}

TASK 2:
 Implement a circuit that returns |01> and |10> with equal probability (50% for each).
Requirements :
  -The circuit should consist only of CNOTs, RXs and RYs. 
  -Start from all parameters in parametric gates being equal to 0 or randomly chosen. 	
  -You should find the right set of parameters using gradient descent (you can use more advanced optimization methods if you like). 
  -Simulations must be done with sampling (i.e. a limited number of measurements per iteration) and noise. 

Compare the results for different numbers of measurements: 1, 10, 100, 1000. 

Bonus question:
How to make sure you produce state |01> + |10> and not |01> - |10> ?

(Actually for more careful readers, the “correct” version of this question is posted below:
How to make sure you produce state  |01⟩  +  |10⟩  and not any other combination of |01> + e(i phi)|10⟩ (for example |01⟩  -  |10⟩)?)
