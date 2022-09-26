# paint part is modified https://gist.github.com/nikhilkumarsingh/85501ee2c3d8c0cfa9d1a27be5781f06

import tkinter as tk
import numpy as np
from PIL import Image
import fcm

class FCMGame(object):

    DEFAULT_PEN_SIZE = 5.0
    DEFAULT_COLOR = 'black'

    def __init__(self, DirichletVoxels, NeumannVoxelsX, NeumannVoxelsY, Material, n):
        self.DirichletVoxels = DirichletVoxels
        self.NeumannVoxelsX = NeumannVoxelsX
        self.NeumannVoxelsY = NeumannVoxelsY
        self.Material = Material
        self.n = n
        
        self.scaleimage = 20
        
        self.root = tk.Tk()

        self.root.title("Maximize the Stiffness")
  
        self.pen_button = tk.Button(self.root, text='pen', command=self.use_pen)
        self.pen_button.grid(row=0, column=0)

        self.eraser_button = tk.Button(self.root, text='eraser', command=self.use_eraser)
        self.eraser_button.grid(row=0, column=1)

        self.choose_size_button = tk.Scale(self.root, from_=1, to=10, orient=tk.HORIZONTAL)
        self.choose_size_button.grid(row=0, column=2)
        
        self.compute_button = tk.Button(self.root, text='compute', command=self.compute)
        self.compute_button.grid(row=0, column=3)
        
        self.clear_button = tk.Button(self.root, text='clear',command=self.clear)
        self.clear_button.grid(row=0, column=4)
        
        self.c = tk.Canvas(self.root, bg='white', width=self.n * self.scaleimage, height=self.n * self.scaleimage)
        self.c.grid(row=1, rowspan=1, columnspan=5)
            
        self.c1 = tk.Canvas(self.root, bg='lightgray', width=self.n * self.scaleimage, height=5 * self.scaleimage)
        self.c1.grid(row=2, columnspan=5)
        
        self.c2 = tk.Canvas(self.root, bg='white', width=self.n * self.scaleimage, height=11 * self.scaleimage)
        self.c2.grid(row=3, columnspan=5)
        self.c2.create_polygon([5*self.scaleimage,2*self.scaleimage,6*self.scaleimage,2*self.scaleimage,
                                6*self.scaleimage,3*self.scaleimage,5*self.scaleimage,3*self.scaleimage],fill="gray")
        self.c2.create_text(25*self.scaleimage,2.5*self.scaleimage, text="Dirichlet Boundary Condition in x and y")
        self.c2.create_polygon([5*self.scaleimage,5*self.scaleimage,6*self.scaleimage,5*self.scaleimage,
                                6*self.scaleimage,6*self.scaleimage,5*self.scaleimage,6*self.scaleimage],fill="red")
        self.c2.create_text(25*self.scaleimage,5.5*self.scaleimage, text="Neumann Boundary Condition in x")
        self.c2.create_polygon([5*self.scaleimage,8*self.scaleimage,6*self.scaleimage,8*self.scaleimage,
                                6*self.scaleimage,9*self.scaleimage,5*self.scaleimage,9*self.scaleimage],fill="blue")
        self.c2.create_text(25*self.scaleimage,8.5*self.scaleimage, text="Neumann Boundary Condition in y")

        self.setup()
        self.root.mainloop()

    def setup(self):
        self.old_x = None
        self.old_y = None
        self.line_width = self.choose_size_button.get()
        self.drawBoundaryConditions()
        self.color = self.DEFAULT_COLOR
        self.eraser_on = False
        self.active_button = self.pen_button
        self.c.bind('<B1-Motion>', self.paint)
        self.c.bind('<ButtonRelease-1>', self.reset)
        self.materialUsed = 0
        
    def use_pen(self):
        self.activate_button(self.pen_button)

    def use_eraser(self):
        self.activate_button(self.eraser_button, eraser_mode=True)

    def activate_button(self, some_button, eraser_mode=False):
        self.active_button.config(relief=tk.RAISED)
        some_button.config(relief=tk.SUNKEN)
        self.active_button = some_button
        self.eraser_on = eraser_mode

    def paint(self, event):
                 
        if self.materialUsed >= 1.:
            self.eraser_on=True
            print("Maximum amount of material is reached. Eraser activated")
        
        self.line_width = self.choose_size_button.get() * self.scaleimage * 2
        paint_color = 'white' if self.eraser_on else self.color
        if self.old_x and self.old_y:
            self.c.create_line(self.old_x, self.old_y, event.x, event.y,
                               width=self.line_width, fill=paint_color,
                               capstyle=tk.ROUND, smooth=False, splinesteps=36)
        self.drawBoundaryConditions()
                    
        self.progressBar()
            
        self.old_x = event.x
        self.old_y = event.y
        
    def progressBar(self):
        fileName = "temp"
        self.c.postscript(file = fileName + '.png')
        img = Image.open(fileName + '.png')
        img = np.sum(np.array(img),axis=2)
        self.materialUsed = (1 - (np.sum(img>0))/len(img)**2) / self.Material 
        progressBar = [0,0,
                       0,5*self.scaleimage,
                       self.n*self.scaleimage*self.materialUsed,5*self.scaleimage,
                       self.n*self.scaleimage*self.materialUsed,0]
        self.c1.create_polygon(progressBar,fill="black")
        invertedProgressBar = [self.n*self.scaleimage*self.materialUsed,0,
                               self.n*self.scaleimage*self.materialUsed, 5*self.scaleimage,
                               self.n*self.scaleimage, 5*self.scaleimage,
                               self.n*self.scaleimage, 0]
        self.c1.create_polygon(invertedProgressBar,fill="lightgray")
        self.c1.create_text(25*self.scaleimage,2.5*self.scaleimage, text="{:.2f}%".format(np.min((self.materialUsed*100, 100))), fill="white")
        
    def compute(self):
        self.compute_button["state"] = "disabled"
        
        # preprocessing        
        image = self.saveAsPNG("temp")        
        indicator = -np.transpose(np.around(np.flip(np.array(image), axis=0)/255))+1

        # disable GUI
        self.root.destroy()
        
        # model parameters
        # discretization
        n = len(indicator)
        alpha = 1e-6 
        
        # boundary condition extraction from voxels
        DirichletBCs = []
        for iVoxel, jVoxel in self.DirichletVoxels:
            DirichletBCs += fcm.eft(iVoxel, jVoxel, n)
        NeumannBCs = []
        for iVoxel, jVoxel in self.NeumannVoxelsX:
            NeumannBCs += np.array(fcm.eft(iVoxel, jVoxel, n))[:8:2].tolist()
        for iVoxel, jVoxel in self.NeumannVoxelsY:
            NeumannBCs += np.array(fcm.eft(iVoxel, jVoxel, n))[1:8:2].tolist()
        
        NeumannBCs = np.transpose(np.concatenate((np.array(NeumannBCs),np.ones(len(NeumannBCs),dtype=int))).reshape((2,len(NeumannBCs)))).tolist()
        
        # material parameters
        E = 100. 
        nu = 0.3
        
        # assembly of fcm problem and solving
        K = fcm.globalStiffnessMatrix(E, nu, indicator, alpha, n)
        K = fcm.applyHomogeneousDirichletBCs(K, DirichletBCs)
        F = fcm.globalForceVectorFromNeumannBCs(NeumannBCs, n)
        U = fcm.solve(K, F)
        
        # postprocessing
        gps = 12 
        
        xi, eta = fcm.getPostProcessingGrid(n, gps)
        Ux, Uy = fcm.getDisplacements(U, n, gps)
        x, y = fcm.getPostProcessingGrid(n, gps)

        compliance = np.transpose(F)@U
        print("Compliance: {:.2e}".format(compliance))
        print("Stiffness: {:.2e}".format(1./compliance))

        fcm.contourPlot(x, y, Ux, indicator, DirichletBCs, NeumannBCs, n, gps, title="Displacement in $x$\n Compliance: {:.2e}\n Stiffness: {:.2e}".format(compliance, 1/compliance))
        fcm.contourPlot(x, y, Uy, indicator, DirichletBCs, NeumannBCs, n, gps, title="Displacement in $y$\n Compliance: {:.2e}\n Stiffness: {:.2e}".format(compliance, 1/compliance))

        self.indicator = indicator
        self.compliance = compliance

    def saveAsPNG(self, fileName):
        self.c.postscript(file = fileName + '.png')
        img = Image.open(fileName + '.png')
        img = img.resize((self.n, self.n), Image.NEAREST).convert("L")
        img.save(fileName + '.png', 'png')
        return img
    
    def drawBoundaryConditions(self):
        bcSquare = np.array([0,self.scaleimage*self.n,
                             self.scaleimage,self.scaleimage*self.n,
                             self.scaleimage,self.scaleimage*(self.n-1),
                             0,self.scaleimage*(self.n-1)])
        bcSquareShifted = np.zeros(8)
        # Dirichlet Boundary Conditions
        for iVoxel in range(len(self.DirichletVoxels)):
            bcSquareShifted[0::2] = bcSquare[0::2] + self.DirichletVoxels[iVoxel,0] * self.scaleimage
            bcSquareShifted[1::2] = bcSquare[1::2] - self.DirichletVoxels[iVoxel,1] * self.scaleimage            
            self.c.create_polygon(bcSquareShifted.tolist(), fill="gray")
        # Neumann Boundary Conditions
        for iVoxel in range(len(self.NeumannVoxelsX)):
            bcSquareShifted[0::2] = bcSquare[0::2] + self.NeumannVoxelsX[iVoxel,0] * self.scaleimage
            bcSquareShifted[1::2] = bcSquare[1::2] - self.NeumannVoxelsX[iVoxel,1] * self.scaleimage    
            self.c.create_polygon(bcSquareShifted.tolist(), fill="red")
        for iVoxel in range(len(self.NeumannVoxelsY)):
            bcSquareShifted[0::2] = bcSquare[0::2] + self.NeumannVoxelsY[iVoxel,0] * self.scaleimage
            bcSquareShifted[1::2] = bcSquare[1::2] - self.NeumannVoxelsY[iVoxel,1] * self.scaleimage    
            self.c.create_polygon(bcSquareShifted.tolist(), fill="blue")    

    def clear(self):
        self.c.delete("all")
        self.drawBoundaryConditions()
        #clear progressbar
        self.materialUsed = 0
        self.progressBar()
        
    def reset(self, event):
        self.old_x, self.old_y = None, None