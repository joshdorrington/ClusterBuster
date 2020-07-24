# What is this for? We want to take in a PC sequence, a jet speed sequence and a Z500 sequence.
#These must match in terms of iris time coord metadata. Once that is checked these funcs should perform all
#aspects of the clustering problem.

#Needed imports and additional functions:
import sys
import numpy as np
import iris
import cf_units
from cluster import Kmeans_cluster, get_cluster_cube, correlate_clusters

#regresses series x from y (i.e. jet speed from pcs)
def regress(x,y, deg=1, prnt=True):
    import numpy as np
    import numpy.ma as ma
    from scipy.stats import pearsonr
    #Check for mask
    if ma.is_masked(x) or ma.is_masked(y):
        fitter = ma.polyfit
    else:
        fitter = np.polyfit
    model = fitter(x,y,deg)
    prediction = np.polyval(model, x)
    residuals = y - prediction
    corr, pval = pearsonr(prediction, y)
    if prnt:
        print("Fitted model: y = %.3f*x + %.3f" % (model[0], model[1]))
        print("Correlation = %.3f (p=%.3f)" % (corr, pval))
        print("Explained variance = %.3f" % (corr**2))
        print("Returning (coefficients, prediction, residuals)")
    return corr, residuals


#Wrapper around regress
def regress_pc(pc,regressor):
    npcs=pc.shape[1]
    output=[regress(regressor,pc[:,i].data,prnt=False) for i in range(npcs)]
    
    corrs=np.array([o[0] for o in output])
    r_data=np.array([o[1] for o in output])
    
    regressed_pc=pc.copy()
    regressed_pc.data=r_data.T
    return corrs,regressed_pc

def make_cube_with_different_1st_axis(cube,new_t,t_ax=None):
    S=cube.shape
    
    if t_ax is None:
        t_ax=iris.coords.DimCoord(np.arange(0,new_t),"time",units=cf_units.Unit(f"days since {cf_units.EPOCH}"))
        
    new_cube=iris.cube.Cube(data=np.zeros([new_t,*S[1:]]))
    
    new_cube.add_dim_coord(t_ax,0)
    for i,coord in enumerate(cube.dim_coords[1:]):
        new_cube.add_dim_coord(coord,i+1)
        
    new_cube.standard_name=cube.standard_name
    new_cube.long_name=cube.long_name
    new_cube.var_name=cube.var_name
    new_cube.units=cube.units
    new_cube.metadata=cube.metadata
    new_cube.attributes=cube.attributes
    
    try:
        new_cube.attributes["history"]=\
        new_cube.attributes["history"]+" Remade into a different shape by function make_cube_with_different_1st_axis."
    except:
        new_cube.attributes["history"]=" Remade into a different shape by function make_cube_with_different_1st_axis."
        
    return new_cube

#The Main Class Object:

class ClusteringExperiment:
    
    def __init__(self,exp_id="Unnamed",pc_cube=None,regressor=None,field_data=None):
        
        self.id=exp_id
        self.pcs=pc_cube
        self.regressor=regressor
        self.field_data=field_data
        
        self._confirm_time_coords_match()
        self.regression_correlations=None
        self.regressed_pcs=None
        self.windowed_pcs=None
        self.windowed_regressed_pcs=None
        self.windowed_field_data=None
        self.windowed_regressor=None
        self.clusters={}
        self.cluster_cubes={}
        self.cluster_correlations={}
        
    #Not currently very rigorous.
    #Makes sure the time axes align.
    def _confirm_time_coords_match(self):
        
        t_axes=[]
        if self.pcs is not None:
            t_axes.append(self.pcs.coord("time"))
        if self.regressor is not None:
            t_axes.append(self.regressor.coord("time"))
        if self.field_data is not None:
            t_axes.append(self.field_data.coord("time"))
            
        for t_ax in t_axes:
            assert np.all(t_ax.points==t_axes[0].points)
            
    #regresses self.regressor against self.pcs
    def regress_pcs(self):
            
        if self.regressor is None:
            raise(ValueError("No regressor attribute defined."))
            
        
        corr,regressed_pc=regress_pc(self.pcs,self.regressor.data)
        
        self.regression_correlations=corr
        self.regressed_pcs=regressed_pc
    
    #Called by combine_with, and used to stick different
    #PC sequences together. 
    def _combine_attribute(self,attr,cluster_array,time_coord=None):
        
        array=[getattr(self,attr).data]
        
        for C in cluster_array:
            array.append(getattr(C,attr).data)
            
        #The atleast_3d here helps make sure 1D attributes
        #(like the regressor) get treated the same way as 2D attributes
        array=np.vstack(np.atleast_3d(array))
        T=array.shape[0]

        new_attr=make_cube_with_different_1st_axis(getattr(self,attr),T,t_ax=time_coord)
        #We want to get rid of length 1 dimensions, hence the squeeze here
        new_attr.data=np.squeeze(array)
        return new_attr
    
    #Combines the current ClusteringExperiment PCs with some others,
    #appending state sequences together. 
    def combine_with(self,cluster_array,time_coord=None,new_id=None):
        
        New_ClusterExperiment=ClusteringExperiment(exp_id=new_id)
        
        for attribute in ["pcs","regressed_pcs","regressor","field_data"]:
            if getattr(self,attribute) is not None:
                combined_attribute=self._combine_attribute(attribute,cluster_array,time_coord)
                setattr(New_ClusterExperiment,attribute,combined_attribute)
                
        return New_ClusterExperiment
    
    def _window_attribute(self,attribute,width,overlap):
        
        data=getattr(self,attribute)
        
        windowed_array=[]
        
        T=data.shape[0]
        window_num=np.floor((T-width)/overlap).astype(int)
        windows=[slice(i*overlap,(i*overlap)+width) for i in range(window_num)]
        
        for window in windows:
            windowed_array.append(data[window])
            
        return windowed_array
    
    def window_data(self,width,overlap):
        
        for attribute in ["pcs","regressed_pcs","regressor","field_data"]:
            if getattr(self,attribute) is not None:
                windowed_attribute=self._window_attribute(attribute,width,overlap)
                setattr(self,"windowed_"+attribute,windowed_attribute)
        
    
    def cluster_pcs(self,Ks,pc_list=None):
        
        self.Ks=Ks
        
        if pc_list is None:
            pc_list=["pcs","regressed_pcs","windowed_pcs","windowed_regressed_pcs"]
            
        for pcs in pc_list:
            if getattr(self,pcs) is not None:
                clusters=self._cluster_attribute(pcs,Ks)
                self.clusters[pcs]=clusters
            
    def _cluster_attribute(self,pcs,Ks):
        
        data=getattr(self,pcs)
        #data will either be an iris cube or a list of cubes. 
        #We try and iterate and if that fails then its a cube
        
        try:
            clusters=[{K:Kmeans_cluster(cube.data,K) for K in Ks} for cube in data]
        except:
            clusters={K:Kmeans_cluster(data.data,K) for K in Ks}
            
        return clusters
    
    def get_cluster_cubes(self,pc_list=None):
        
        if pc_list is None:
            pc_list=["pcs","regressed_pcs","windowed_pcs","windowed_regressed_pcs"]
                    
        for pc in pc_list:
            
            if getattr(self,pc) is not None:
                
                cl_data=self.clusters[pc]
                #Assume its iterable (i.e. windowed):
                try:
                    ccs=[{K:get_cluster_cube(F,cl[K].states) for K in self.Ks} for F,cl in zip(self.windowed_field_data,cl_data)]
                #If not, then its full data:
                except:
                    
                    ccs={K:get_cluster_cube(self.field_data,cl_data[K].states) for K in self.Ks}
                    
                self.cluster_cubes[pc]=ccs
        
    def _reorder_clusters(self,pc,K,mapping,window=None):
        
        order=np.array([m[1] for m in mapping])
        
        if window is None:
            #Reorder cluster cubes
            self.cluster_cubes[pc][K]=self.cluster_cubes[pc][K][order]
            #Reorder cluster data
            self.clusters[pc][K].reorder(mapping)
        else:
            #Reorder cluster cubes
            self.cluster_cubes[pc][window][K]=self.cluster_cubes[pc][window][K][order]
            #Reorder cluster data
            self.clusters[pc][window][K].reorder(mapping)

        
    def correlate_clusters(self,reference_clusters,reorder_clusters=False,pc_list=None,reference_id="None"):
        
        if pc_list is None:
            pc_list=["pcs","regressed_pcs","windowed_pcs","windowed_regressed_pcs"]

        correlation_dict={}
        
        for pc in pc_list:
            
            cubes1=self.cluster_cubes[pc]
            
            
            #Non windowed clusters:
            if type(cubes1) is dict:
                
                mean_corr_dict={}
                reg_corr_dict={}
                for K in self.Ks:
                
                    (mean_corr,reg_corrs),mapping=correlate_clusters(reference_clusters[K],cubes1[K],and_mapping=True,mean_only=False)
                    
                    mean_corr_dict[K]=mean_corr
                    reg_corr_dict[K]=reg_corrs
                    
                    if reorder_clusters:
                        self._reorder_clusters(pc,K,mapping,window=None)
                        
                correlation_dict[pc]=[mean_corr_dict,reg_corr_dict]
                
            #Windowed clusters
            elif type(cubes1) is list:
                
                mean_corr_arr=[]
                reg_corr_arr=[]
                
                for w,cc1 in enumerate(cubes1):
                    
                    mean_corr_dict={}
                    reg_corr_dict={}
                    for K in self.Ks:

                        (mean_corr,reg_corrs),mapping=correlate_clusters(reference_clusters[K],cc1[K],and_mapping=True,mean_only=False)

                        mean_corr_dict[K]=mean_corr
                        reg_corr_dict[K]=reg_corrs

                        if reorder_clusters:
                            self._reorder_clusters(pc,K,mapping,window=w)
                            
                    mean_corr_arr.append(mean_corr_dict)
                    reg_corr_arr.append(reg_corr_dict)
                    
                correlation_dict[pc]=[mean_corr_arr,reg_corr_arr]

            else:
                raise(ValueError("cubes should be stored either as a dict or list"))
            
        self.cluster_correlations[reference_id]=correlation_dict
        
    #Just a little syntactic sugar to make data retrieval more convenient
    
    def get_cl_data(self,pc_data,cl_attr,K,window=None):
        if window is None:
            return getattr(self.clusters[pc_data][K],cl_attr)
        else:
            return getattr(self.clusters[pc_data][window][K],cl_attr)
        
    def get_cc_data(self,pc_data,K,window=None):
        if window is None:
            return self.cluster_cubes[pc_data][K]
        else:
            return self.cluster_cubes[pc_data][window][K]
        
    def get_correlation_data(self,exp_id,pc_data,K,window):
        
        if window is None:
            return self.cluster_correlations[exp_id][pc_data][K]
        else:
            return self.cluster_correlations[exp_id][pc_data][window][K]

    
#Workflow:
run=False
if run:
    test_pcs1=iris.load_cube("../data/derived_data/primavera_data/pcs/EC-Earth3P-HR_r1i1p2f1_pcs.nc")
    test_js1=iris.load_cube("../data/derived_data/primavera_data/jet_speeds/EC-Earth3P-HR_r1i1p2f1_jet_speed.nc")
    test_z5001=load_pickle("../data/raw_data/primavera/u_full_Z500.pkl")["EC-Earth3P-HR_r1i1p2f1"]
    
    test_pcs2=iris.load_cube("../data/derived_data/primavera_data/pcs/EC-Earth3P-HR_r2i1p2f1_pcs.nc")
    test_js2=iris.load_cube("../data/derived_data/primavera_data/jet_speeds/EC-Earth3P-HR_r2i1p2f1_jet_speed.nc")
    test_z5002=load_pickle("../data/raw_data/primavera/u_full_Z500.pkl")["EC-Earth3P-HR_r2i1p2f1"]
    
    T=30*90
    t=10*90
    
    C=ClusteringExperiment(exp_id="test1",pc_cube=test_pcs1,regressor=test_js1,field_data=test_z5001)
    C.regress_pcs()

    C2=ClusteringExperiment(exp_id="test2",pc_cube=test_pcs2,regressor=test_js2,field_data=test_z5002)
    C2.regress_pcs()

    Ccomb=C.combine_with([C2],time_coord=None,new_id="joined_test")

    Ccomb.window_data(width=T,overlap=t)

    Ccomb.cluster_pcs(Ks=np.arange(2,4))

    Ccomb.get_cluster_cubes()

    Ccomb.correlate_clusters(Ccomb.cluster_cubes["pcs"],reorder_clusters=True,pc_list=["pcs","windowed_pcs"])

    print("done.")