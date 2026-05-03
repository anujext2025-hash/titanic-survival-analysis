import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
data = {
    'dept':    ['HR', 'Tech', 'Finance', 'Marketing', 'Sales'],
    'salary':  [50000, 110000, 59000, 72000, 65000],
    'headcount': [3, 8, 2, 4, 5]
}
df = pd.DataFrame(data)

#barchart
plt.figure(figsize=(10,5))
plt.bar(df['dept'], df['salary'], color='pink')
plt.title('salary by department')
plt.xlabel('department')
plt.ylabel('salary')
plt.tight_layout()
#plt.show()

#subplot
fig, axes = plt.subplots(1,2, figsize=(12, 4))

axes[0].plot(df['dept'], df['salary'], color='skyblue')
axes[0].set_title('salary by department')
axes[0].set_xlabel('Department')

axes[1].plot(df['dept'], df['headcount'], color='pink')
axes[1].set_title('headcount by department')
axes[1].set_xlabel('Department')

plt.tight_layout()
#plt.show()

#linechart
months = ['Jan','Feb','Mar','Apr','May','Jun']
sales  = [3000, 3500, 2800, 4200, 5000, 4800]

plt.figure(figsize=(10,5))
plt.plot(months,sales , marker='o', color='pink', linewidth=2)
plt.title('sales per month')
plt.xlabel('month')
plt.ylabel('sales')
plt.tight_layout()
#plt.show()

#histogram
ages = np.random.normal(30, 8, 200)
plt.figure(figsize=(10,6))
plt.hist(ages, bins=20, color='pink', edgecolor='black')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.tight_layout()
#plt.show()

#SEABORN----
