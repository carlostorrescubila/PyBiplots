"""This ."""

# ----------------------------------
# Name        : HJ_Biplot.py
# Author      : C. Torres-Cubilla
# Contact     : carlos_t22@usal.es
# ----------------------------------


## Libraries
import numpy as np 
import pandas as pd
import scipy.linalg as la
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from sklearn.preprocessing import StandardScaler
from adjustText import adjust_text

class fit(object):
    
    def __init__(self, X, Transform = 'Standardize columns'):
        
            
        """Fit HJ-Biplot.
         
        Parameters
        ----------
        X : array-like
            Data to fit the biplot method.
        Transform : str
            Character indicating the data transforming. Allowed values are None, 
            'Row centering', 'Standardize rows', 'Column centering', 'Standardize columns'.
        
        Returns
        -------
        X : array-like
            Data used withou transformation.
        Transformation : chr
            Transformation used before the fitting.
        eigenvalues : array-like
            Eigenvalues obtained
        explained_variance : array-like
            Explained variance for each of the dimension.
        row_coordinates : array-like
            Coordinates of the rows in each new dimension.
        column_coordinates : array-like
            Coordinates of the columns in each new dimension.
        row_contributions : array-like
            Contributions of each row to the new dimensions. 
        column_contributions : array-like
            Contributions of each column to the new dimensions.
            
        Examples
        --------
        >>> # Load example data
        >>> X = pd.DataFrame(data=load_iris().data, columns=load_iris().feature_names)
        >>> # Fit biplot
        >>> model = hj.fit(X, Transform='Standardize columns')
        >>> # Results
        >>> # >Eigenvalues
        >>> model.eigenvalues
        >>> # >Variance explained
        >>> model.explained_variance.round(2)
        >>> # >Loadings
        >>> model.loadings
        >>> # >Coordinates
        >>> # >>Rows
        >>> model.row_coordinates.round(2)
        >>> # >> columns
        >>> model.column_coordinates.round(2)
        >>> # >Contributions
        >>> # >>Rows
        >>> model.row_contributions
        >>> # >>Columns
        >>> model.column_contributions
        >>> # Plot
        >>> model.plot(groups=load_iris().target, ind_name=False)
        """
        
        self.X = X
        self.Transformation = Transform
            
        ##### >Transform data #####
        if Transform == 'Column centering':
            scaler = StandardScaler(with_std = False)
            X = pd.DataFrame(scaler.fit_transform(X), 
                             columns = X.columns, 
                             index = X.index)
        elif Transform == 'Row centering':
            scaler = StandardScaler(axis = 1, with_std = False)
            X = pd.DataFrame(scaler.fit_transform(X), 
                             columns = X.columns, 
                             index = X.index)
        elif Transform == 'Standardize columns':
            scaler = StandardScaler()
            X = pd.DataFrame(scaler.fit_transform(X), 
                             columns = X.columns, 
                             index = X.index)
        elif Transform == 'Standardize rows':
            scaler = StandardScaler(axis = 1)
            X = pd.DataFrame(scaler.fit_transform(X), 
                             columns = X.columns, 
                             index = X.index)

        ##### >Names of the axis #####
        axis_tag = list()
        for i in list(range(min(X.shape))):    
            axis = 'Axis ' + str(i+1)
            axis_tag = np.concatenate((axis_tag, axis), axis=None)

        ##### >Singular value decomposition #####
        U, d, V_t = np.linalg.svd(X, full_matrices = False)
        eigvals, eigvecs = la.eig(X.corr().values)
        self.eigenvalues = eigvals.real
        self.loadings = pd.DataFrame(V_t.T, 
                                    index = X.columns, 
                                    columns = axis_tag)
        D = np.diag(d)

        ##### >Explained variance #####
        explained_variance = (d**2)/sum(d**2) * 100
        self.explained_variance = explained_variance

        ##### >Coordinates #####
        ### >>Rows ###
        self.row_coordinates = pd.DataFrame(
            X.values @ V_t.T, 
            index = X.index, 
            columns = axis_tag
        )
        ### >>Columns ###
        self.column_coordinates = pd.DataFrame(
            V_t.T @ D, 
            index = X.columns, 
            columns = axis_tag
        )

        ##### >Contributions #####
        ### >>Rows ###
        rc_2 = self.row_coordinates**2
        sum_rc_2 = rc_2.sum(axis=1)
        row_contributions = (rc_2.T/sum_rc_2).T * 1000
        self.row_contributions = pd.DataFrame(
            row_contributions.values.astype(int), 
            index = X.index, 
            columns = axis_tag
        )
        ### >>Columns ###
        cc_2 = self.column_coordinates**2
        sum_cc_2 = cc_2.sum(axis=1)
        column_contributions = (cc_2.T/sum_cc_2).T*1000
        self.column_contributions = pd.DataFrame(
            column_contributions.values.astype(int), 
            index = X.columns, 
            columns = axis_tag
        )
            
    def plot(self, axis = (1,2), groups = None, palette = None, convex_hull = False, ind_name = True, vec_name = True, 
             vec_color = 'black', name_fontsize = 13, axis_fontsize = 20, angle_vec_name = True, adjust_ind_name = False,
             legend='brief', arrow_head = 0.2): 
        
        """Draw the Biplot.
        
        Description
        -----------
        Plot two dimensions with the rows and columns considered.
                
        Parameters
        ----------
        axis : (int, int), default: (1,2)
            The x and y axis to be plotted.
        groups : array-like, optional, default: None
            Array which contains the groups to use.
        palette : Bool, optional, default: None
            Palette to use for points.
        convex_hull : Bool, default: True
            Show the convex hull for groups.
        ind_name : Bool, default: True
            Print the index of X
        vec_name : Bool, default: True
            Print the columns of X
        vec_color : chr, default: 'black'
            Color of the arrows.
        name_fontsize : int, default:13
            Size of the tags of points and arrows. 
        axis_fontsize : int, default: 20
            Size of the tags of labels
        angle_vec_name : Bool, default: True
            Angle of the arrow tags. If True the tags get the same angle of the arrow. 
            If  False the tags get Null angle. 
        adjust_ind_name : Bool, default: False
            If true it will adjust the point tags to don't get overlapped.
        legend : chr, optional, default:'brief'
            How to draw the legend. If “brief”, numeric groups and size variables will 
            be represented with a sample of evenly spaced values. If “full”, every 
            group will get an entry in the legend. If False, no legend data is added 
            and no legend is drawn.
        arrow_head : float, default: 0.2
            size of the arrow head.
            
        Returns
        -------
        tuple containing (fig, ax)
        """
        
        X = self.X
        ind = self.row_coordinates
        vec = self.column_coordinates

        #Name of the variables
        vec_tag = X.columns  
        
        axis_x = axis[0]-1
        axis_y = axis[1]-1

        #Limits of the figure
        x_min = min(min(ind.iloc[:,axis_x]), min(vec.iloc[:,axis_x])) - arrow_head/0.15
        y_min = min(min(ind.iloc[:,axis_y]), min(vec.iloc[:,axis_y])) - arrow_head/0.15
        x_max = max(max(ind.iloc[:,axis_x]), max(vec.iloc[:,axis_x])) + arrow_head/0.15
        y_max = max(max(ind.iloc[:,axis_y]), max(vec.iloc[:,axis_y])) + arrow_head/0.15
        
        #No group parmeters
        if groups is None:
            groups = [1] * ind.shape[0]
            legend = False
        else: groups = groups
        
        #Palette by default
        if palette is None:
            palette = 'Set1'
            
        #Figure
        ax = sns.scatterplot(x = ind.iloc[:,axis_x], 
                             y = ind.iloc[:,axis_y], 
                             data = ind, 
                             hue = groups, 
                             palette = palette, 
                             zorder = 2, 
                             legend = legend)
        plt.xlim([x_min, x_max]) 
        plt.ylim([y_min, y_max])
        plt.axvline(0, color ='silver', zorder = 1)
        plt.axhline(0, color ='silver', zorder = 1)    
        plt.xlabel(ind.columns[axis_x] + ' (' + str(round(self.explained_variance[axis_x], 2)) + '%)' , 
                   fontsize = axis_fontsize, 
                   color = 'black')
        plt.ylabel(ind.columns[axis_y] + ' (' + str(round(self.explained_variance[axis_y], 2)) + '%)' , 
                   fontsize = axis_fontsize, 
                   color = 'black') 


        # Print the vectors of each variable
        rep = 0
        i = 0
        for i, vector in enumerate(vec):
            ax.arrow(0, 0, 
                     vec.iloc[i,axis_x], 
                     vec.iloc[i,axis_y], 
                     head_width = arrow_head, 
                     head_length = arrow_head,
                     color = vec_color, 
                     zorder = 3)        
            i = i+1
            rep = rep+1

        # Print the tags of each individue
        if ind_name == True: 
            x = ind.iloc[:,0]
            y = ind.iloc[:,1]
            if adjust_ind_name == False:
                for i in range(ind.shape[0]):
                    ax.text(x[i], y[i], 
                            ind.index[i],  
                            color = ax.collections[0].get_facecolors()[i], 
                            fontsize = name_fontsize , 
                            zorder = 2)
            if adjust_ind_name == True: 
                text = [plt.text(x[i], y[i], 
                            ind.index[i],  
                            color = ax.collections[0].get_facecolors()[i], 
                            fontsize = name_fontsize , 
                            zorder = 2) for i in range(ind.shape[0])]
                adjust_text(text)
        
        # Convex Hull
        c_p = pd.DataFrame(ax.collections[0].get_facecolors(), index = ind.index)
        if convex_hull == True:
            from scipy.spatial import ConvexHull
            points_all_ = ind.iloc[:,:2]
            points_all_ = points_all_.assign(groups = groups)
            points_all = pd.concat([points_all_, c_p], axis = 1)
            points_group_i = points_all.groupby('groups')
            for group_i in points_group_i.groups:
                points_tags = points_all.loc[points_group_i.groups[group_i], :]
                points = points_tags.iloc[:,:2]
                if points.shape[0] == 1:
                    year = 2019 #This don´t do anything
                else:
                    hull = ConvexHull(points)
                    for simplex in hull.simplices:
                        plt.plot(points.iloc[simplex, 0], 
                                 points.iloc[simplex, 1], 'k-', 
                                 color = points_tags.iloc[0,3:], 
                                 alpha = 0.25)

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
                    x = x + (arrow_head)
                    if y > 0:
                        y = y + (arrow_head)
                    else:
                        y = y - (arrow_head)
                    ax.text(x, y, vec_tag,  
                            color = vec_color, 
                            fontsize = name_fontsize, 
                            horizontalalignment = 'left', 
                            verticalalignment = 'center', 
                            rotation = angle, 
                            rotation_mode = 'anchor', 
                            name = 'serif')                        
                else:
                    x = x - (arrow_head)
                    if y > 0:
                        y = y + (arrow_head)
                    else:
                        y = y - (arrow_head)
                    ax.text(x, y, vec_tag,  color = vec_color, fontsize = name_fontsize, 
                            horizontalalignment = 'right', verticalalignment = 'center', 
                            rotation = angle, rotation_mode = 'anchor', name = 'serif')         
                i = i+1
                rep = rep+1

        plt.tight_layout()    