import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

x = np.linspace(0, 20, 250)
x2 = np.linspace(-10, 10, 250)

fig = plt.figure(figsize=(16,5), dpi=100)
grid_pos = gridspec.GridSpec(2, 4, width_ratios=[2,2,2,2], height_ratios=[1,1])
div_x =     fig.add_subplot(grid_pos[0,0])
sqrt =      fig.add_subplot(grid_pos[1,0])
squared =   fig.add_subplot(grid_pos[0,1])
squaredodd = fig.add_subplot(grid_pos[1,1])
yequalsx =  fig.add_subplot(grid_pos[0,2])
cos =       fig.add_subplot(grid_pos[1,2])
sin =       fig.add_subplot(grid_pos[0,3])
tan =       fig.add_subplot(grid_pos[1,3])

div_x.plot      (x, [1/i for i in x], label="1/x")
div_x.legend()
sqrt.plot       (x, np.sqrt(x), label="Square root")
sqrt.legend()
squared.plot    (x2, np.square(x2), label="Squared even (U-shape) (power of)")
squared.legend()
squaredodd.plot (x2, x ** 3, label="Squared odd (S-shape) (power of)")
squaredodd.legend()
yequalsx.plot   (x, [i for i in x], label="y = x")
yequalsx.legend()

# TRIGONOMETRI, läran om sambanden mellan vinklarna och triangelns sidor:
cos.plot        (x, [np.cos(i) for i in x], label="Cosinus")
cos.legend()
sin.plot        (x, [np.sin(i) for i in x], label="Sinus")
sin.legend()
tan.plot        (x, [np.tan(i) for i in x], label="Tangens")
tan.legend()

plt.ylim(0,20)
plt.xlim(0,20)
plt.tight_layout()
plt.show()