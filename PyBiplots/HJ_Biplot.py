import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from sklearn.preprocessing import StandardScaler

class fit(object):
    
    def __init__(self, X, scale='Standardize columns'):
        
            self.X = X
            self.scale = scale
            
            #Scale the data
            ind_tag = X.index
            self.ind_tag = ind_tag
            vec_tag = X.columns
            self.vec_tag = vec_tag
            if scale == 'Column centering':
                scaler = StandardScaler(with_std = False)
                X = pd.DataFrame(scaler.fit_transform(X), columns=vec_tag, index=ind_tag)
                arrow_head = (X.std()).mean()/10
                text_dist = (X.std()).mean()/20
                lim_plus = (X.std()).mean()*1.5
            elif scale == 'Row centering':
                scaler = StandardScaler(axis = 1, with_std = False)
                X = pd.DataFrame(scaler.fit_transform(X), columns=vec_tag, index=ind_tag)
                arrow_head = (X.std()).mean()/10
                text_dist = (X.std()).mean()/20
                lim_plus = (X.std()).mean()*1.5
            elif scale == 'Standardize columns':
                scaler = StandardScaler()
                X = pd.DataFrame(scaler.fit_transform(X), columns=vec_tag, index=ind_tag)
                arrow_head = 0.1
                text_dist = (X.std()).mean()/20
                lim_plus = (X.std()).mean()*1.5
            elif scale == 'Standardize rows':
                scaler = StandardScaler(axis = 1)
                X = pd.DataFrame(scaler.fit_transform(X), columns=vec_tag, index=ind_tag)
                arrow_head = 0.1
                text_dist = (X.std()).mean()/20
                lim_plus = (X.std()).mean()*1.5
            self.arrow_head = arrow_head
            self.text_dist = text_dist
            self.lim_plus = lim_plus
            
            # Names of the axis 
            axis_tag = list()
            for i in list(range(min(X.shape))):    
                axis = 'Axis ' + str(i+1)
                axis_tag = np.concatenate((axis_tag, axis), axis=None)
            self.axis_tag = axis_tag
                
            #Singular value decomposition
            U, d, V_t = np.linalg.svd(X, full_matrices = False)
            self.eigenvalues = d
            self.weights = pd.DataFrame(V_t.T, index=vec_tag, columns=axis_tag)
            D = np.diag(d)

            #Explained variance
            exp_var = (d**2)/sum(d**2) * 100
            self.exp_var = exp_var

            #Coordinates
            ind = X @ V_t.T
            self.coord_ind = pd.DataFrame(ind.values, index=ind_tag, columns=axis_tag)
            vec = pd.DataFrame(V_t.T @ D, index=vec_tag, columns=axis_tag)
            self.coord_var = vec
            
    def plot(self, axis = (1,2), groups = None, palette = None, vec_color = 'black', 
             ind_name = True, vec_name = True, angle_vec_name = True, legend='brief'): 
        
        X = self.X
        exp_var = self.exp_var
        ind = self.coord_ind
        vec = self.coord_var
        
        #Tags of the individue
        ind_tag = X.index

        #Name of the variables
        vec_tag = X.columns  
        
        axis_x = axis[0]-1
        axis_y = axis[1]-1

        #Limits of the figure
        x_min = min(min(ind.iloc[:,axis_x]), min(vec.iloc[:,axis_x])) - self.lim_plus
        y_min = min(min(ind.iloc[:,axis_y]), min(vec.iloc[:,axis_y])) - self.lim_plus
        x_max = max(max(ind.iloc[:,axis_x]), max(vec.iloc[:,axis_x])) + self.lim_plus
        y_max = max(max(ind.iloc[:,axis_y]), max(vec.iloc[:,axis_y])) + self.lim_plus
        
        #No group parmeters
        if groups is None:
            groups = [1] * ind.shape[0]
            legend = False
        else: groups = groups
            
        #Figure
        fig, ax = plt.subplots()
        sns.scatterplot(x = ind.iloc[:,axis_x], y = ind.iloc[:,axis_y], data = ind, 
                             hue = groups, palette = palette, zorder = 2, legend = legend)
        plt.xlim([x_min, x_max]) 
        plt.ylim([y_min, y_max])
        plt.axvline(0, color ='silver', zorder = 1)
        plt.axhline(0, color ='silver', zorder = 1)    
        plt.xlabel(ind.columns[axis_x] + ' (' + str(round(exp_var[axis_x], 2)) + '%)' , fontsize = 20, color = 'black')
        plt.ylabel(ind.columns[axis_y] + ' (' + str(round(exp_var[axis_y], 2)) + '%)' , fontsize = 20, color = 'black') 


        #Print the vectors of each variable
        rep = 0
        i = 0
        for i, vector in enumerate(vec):
            ax.arrow(0, 0, vec.iloc[i,axis_x], vec.iloc[i,axis_y], head_width = self.arrow_head, head_length = self.arrow_head,
                     color = vec_color, zorder = 3)        
            i = i+1
            rep = rep+1

        #Print the tags of each individue
        if ind_name == True: 
            for i in range(ind.shape[0]):
                x = ind.iloc[:,0]
                y = ind.iloc[:,1]
                ax.text(x[i], y[i], ind.index[i],  color = ax.collections[0].get_facecolors()[i], 
                        fontsize=13 , zorder = 2)
                
         #Print the tags of each vector
        import math
        if vec_name == True:
            rep = 0
            i = 0
        for i, vec_tag in enumerate(vec_tag):
                x = vec.iloc[:, axis_x][i]
                y = vec.iloc[:, axis_y][i]
                if angle_vec_name == True:
                    angle = math.degrees(math.atan(y/x))
                else: 
                    angle = 0
                if x > 0:
                    x = x + (2*self.text_dist)
                    if y > 0:
                        y = y + (2*self.text_dist)
                    else:
                        y = y - (2*self.text_dist)
                    ax.text(x, y, vec_tag,  color = vec_color, fontsize=13, 
                            horizontalalignment = 'left', verticalalignment = 'center', 
                            rotation = angle, rotation_mode = 'anchor', name = 'serif')                        
                else:
                    x = x - (2*self.text_dist)
                    if y > 0:
                        y = y + (2*self.text_dist)
                    else:
                        y = y - (2*self.text_dist)
                    ax.text(x, y, vec_tag,  color = vec_color, fontsize=13, 
                            horizontalalignment = 'right', verticalalignment = 'center', 
                            rotation = angle, rotation_mode = 'anchor', name = 'serif')         
                i = i+1
                rep = rep+1

        plt.tight_layout()    
        
    def eigenvalues(self):
        return self.eigenvalues
    
    def explained_variance(self):
        return self.exp_var
    
    def weights(self):
        return self.weights
    
    def coord_ind(self):
        return self.coord_ind
    
    def coord_var(self):
        return self.coord_var

    def rows_contributions(self):
        rc_2 = self.coord_ind**2
        sum_rc_2 = rc_2.sum(axis=1)
        rows_contributions = round((rc_2.T/sum_rc_2).T*1000).astype(int)
        return pd.DataFrame(rows_contributions.values, index=self.ind_tag, columns=self.axis_tag)
    
    def columns_contributions(self):
        cc_2 = self.coord_var**2
        sum_cc_2 = cc_2.sum(axis=1)
        return round((cc_2.T/sum_cc_2).T*1000).astype(int)