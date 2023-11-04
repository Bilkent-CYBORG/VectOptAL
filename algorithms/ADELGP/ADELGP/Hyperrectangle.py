import numpy as np
import itertools
import matplotlib.pyplot as plt
"""
Represents a hypercube context region.
"""


class Hyperrectangle:
    def __init__(self, lower, upper):
        self.upper = upper
        self.lower = lower
        
        self.dim = len(self.upper)
        self.diameter = np.linalg.norm(np.array(upper)-np.array(lower))

        self.A = self.setA()
        self.b = self.setb()

    def setA(self):
        A = np.vstack((np.eye(self.dim), -np.eye(self.dim)))
        return A


    def setb(self):
        b = np.hstack((np.array(self.lower), -np.array(self.upper)))
        return b


    def intersect(self, rect, t = None):
        lower_new = []
        upper_new = []

        if self.check_intersect(rect): #if the two rectangles overlap
            # with open("nonoverlapvsoverlap.txt", "a") as text_file:
            #     text_file.write("1")

            #return rect #EXPERIMENT

            for l1, l2 in zip(self.lower, rect.get_lower()):
                lower_new.append(max(l1, l2))

            for u1, u2 in zip(self.upper, rect.get_upper()):
                upper_new.append(min(u1, u2))

            return Hyperrectangle(lower_new, upper_new)
        
        else:
                                     # if there is no intersection,then use the new hyperrectangle
            print("NO OVERLAP")
            # with open("nonoverlapvsoverlap.txt", "a") as text_file:
            #     text_file.write("0")

            
            for l1, l2 in zip(self.lower, rect.get_lower()):
                lower_new.append(min(l1, l2))

            for u1, u2 in zip(self.upper, rect.get_upper()):
                upper_new.append(max(u1, u2))

            return Hyperrectangle(lower_new, upper_new)
           # return rect #EXPERIMENT
            #return rect
            
            
            #v1 = self.get_vertices()
            #v2 = rect.get_vertices()
            #plt.plot(v1[:,0],v1[:,1],"r+")
            #plt.plot(v2[:,0],v2[:,1],"b+")
            #plt.show()
            

            #ALG1: Take the average of the means, use the smallest diameter.

            """ vert_self = self.get_vertices()
            vert_rect = rect.get_vertices()

            center_self = np.mean(vert_self,axis=0)
            center_rect = np.mean(vert_rect,axis=0)
            new_mean = (center_self+center_rect)/2

            if self.diameter > rect.diameter:
                return Hyperrectangle(new_mean-(rect.upper-rect.lower)/2,new_mean+(rect.upper-rect.lower)/2)
            else:
                return Hyperrectangle(new_mean-(self.upper-self.lower)/2,new_mean+(self.upper-self.lower)/2) """


            #ALG2: Same as ALG1 but use the round as weight of the old rectangle.
            """ vert_self = self.get_vertices()
            vert_rect = rect.get_vertices()

            center_self = np.mean(vert_self,axis=0)
            center_rect = np.mean(vert_rect,axis=0)
            new_mean = (center_self*t + center_rect)/(t+1)

            if self.diameter > rect.diameter:
                return Hyperrectangle(new_mean-(np.array(rect.upper)-np.array(rect.lower))/2,new_mean+(np.array(rect.upper)-np.array(rect.lower))/2)
            else:
                return Hyperrectangle(new_mean-(np.array(self.upper)-np.array(self.lower))/2,new_mean+(np.array(self.upper)-np.array(self.lower))/2)
 """
            #ALG3:
            vert_rect = rect.get_vertices()
            center_rect = np.mean(vert_rect,axis=0)
            if self.diameter > rect.diameter:
                return Hyperrectangle(center_rect-(np.array(rect.upper)-np.array(rect.lower))/2,center_rect+(np.array(rect.upper)-np.array(rect.lower))/2)
            else:
                return Hyperrectangle(center_rect-(np.array(self.upper)-np.array(self.lower))/2,center_rect+(np.array(self.upper)-np.array(self.lower))/2)




    def get_lower(self):
        return np.array(self.lower)

    def get_upper(self):
        return np.array(self.upper)


    def __str__(self):
        return "Upper: " + str(self.upper) + ", Lower: " + str(self.lower)


    def get_vertices(self):
        a = [[l1, l2] for l1, l2 in zip(self.lower, self.upper)]
        vertex_list = [element for element in itertools.product(*a)]
        return np.array(vertex_list)

    def check_intersect(self,rect): 
        boolean = True
        m = len(self.lower)
        for i in range(m):
             boolean = (boolean and self.lower[i] < rect.upper[i] and self.upper[i] > rect.lower[i])
        return boolean