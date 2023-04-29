import warnings
warnings.filterwarnings('ignore')
                        
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import statsmodels.api as sm 
from scipy.stats import shapiro,normaltest
import sweetviz as sv
import ydata_profiling
from sklearn.preprocessing import LabelEncoder, OneHotEncoder,OrdinalEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer,QuantileTransformer
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder,OrdinalEncoder
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.decomposition import PCA

# Models
from sklearn.linear_model import LogisticRegressionCV , LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier,BaggingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

import xgboost as xgb
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF


from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import  accuracy_score,recall_score, f1_score, precision_score, roc_auc_score


def eda_report_html(df):
    report = sv.analyze(df)
    report.show_html()

def eda_profiling_report(df):
    profile = ydata_profiling(df, minimal=True, title="Profiling Report")
    profile.to_notebook_iframe()

def draw_boxplots(df,variables=['Area','Emayor','Emenor','Exc','Vol','Rat','Perim'],title="Diagramas de cajas"):
    """ Genera los diagramas de cajas de cada variable predictora"""
    fig, axes = plt.subplots(1,7,  figsize=(30, 6))
    fig.suptitle(title, fontsize=16)

    for index, var_name in enumerate(variables):
        sns.boxplot(ax = axes[index], data = df, x='Clase', y=var_name, notch=True)
        axes[index].set_title(df[var_name].name)
        sns.stripplot(ax = axes[index],x='Clase', y=var_name,data = df,alpha = 0.3)
        
def draw_kdeplots(df, variables=['Area','Emayor','Emenor','Exc','Vol','Rat','Perim'],title="Probability distribution "):
    """ Genera los diagramas de (KDE) de cada variable predictora."""
    fig, axes = plt.subplots(1, 7,  figsize=(30, 5))
    fig.suptitle(title,fontsize=16)

    for index, var_name in enumerate(variables):
        sns.kdeplot(ax = axes[index], data= df, x=var_name, hue='Clase', shade=True)
        axes[index].set_title(df[var_name].name)
        
def draw_histplots(df, variables=['Area','Emayor','Emenor','Exc','Vol','Rat','Perim'], title="Histograms plot"):
    """ Genera los histogramas de cada variable predictora"""
    fig, axes = plt.subplots(1, 7,  figsize=(30, 6))
    fig.suptitle(title,fontsize=16)

    for index, var_name in enumerate(variables):
        sns.histplot(ax = axes[index], data= df,  x=var_name, hue='Clase',kde=True)
        axes[index].set_title(df[var_name].name)

def draw_countplot(df, var='Clase',title="Diagrama de barras de la variable {}"):
    """ Diagrama de barras"""
    plt.figsize=(10, 5)
    plt.title(title.format(var))
    sns.countplot(x=df[var],saturation=.65)

def draw_pairplot(df, title='Pair plot', style=["style1","style2","style3"]):
    """ Genera un scatter matrix plot (bivariante)"""
    plt.figsize=(20, 5)
    plt.suptitle(title, fontsize=16)

    if style == "style1": # estilo 1
        sns.pairplot(df , diag_kind="hist", hue="Clase", markers=["o", "s"])
    elif style == "style2":# estilo2
        g = sns.PairGrid(df, hue="Clase")
        g.map_upper(sns.scatterplot)
        g.map_lower(sns.kdeplot )
        g.map_diag(sns.kdeplot, lw=3, legend=False)
    else: # estilo 3
        sns.pairplot(df[['Area','Emayor','Emenor','Exc','Vol','Rat','Perim','Clase']], hue="Clase", kind="reg")

def draw_scatter_plot(data, var_name="Emayor"):
    """ Genera un scatter plot (univariante)""" 
    plt.figure(figsize=(20,6))
    df = data.copy()
    df['Id'] = df.index + 1
    ax = sns.scatterplot(data=df, x="Id", y=var_name, hue="Clase")
    ax.set_title("Distrbución de los puntos de la variable predictora {}".format(var_name), fontsize=16)

def draw_rel_plot(df, var_name1="Emayor", var_name2="Emenor"):
    """ Genera un rel plot (bivariante)""" 
    title= "Distrbución de los puntos de la variable predictora {} y {}".format(var_name1,var_name2)
    plt.figure(figsize=(20,6))
    plt.suptitle(title)
    ax = sns.relplot(data=df,x=var_name1, y=var_name2, hue="Clase", col="Clase")
    plt.show()

def draw_lm_plot(df, var_name1="Emayor", var_name2="Emenor"):
    """ "Generar la lm plot"""
    plt.figure(figsize=(20,6))
    sns.lmplot(x=var_name1, y=var_name2, hue="Clase", markers=["o", "s"], data=df)
    plt.show()

def draw_qq_plots(df,variables=['Area','Emayor','Emenor','Exc','Vol','Rat','Perim'], title='Q-Q Plot'):
    """ Generar la normal q-q plots"""
    fig, axes = plt.subplots(1, 7,  figsize=(30, 5))
    fig.suptitle(variables, fontsize=16)

    for index, var_name in enumerate(variables):
        sm.qqplot(ax = axes[index], data=df[var_name], line="45")
        axes[index].set_title(df[var_name].name)

def draw_pcoordinates_plot(data, variables=['Area','Emayor','Emenor','Exc','Vol','Rat','Perim'], title="Parallel Coordinates plot", n_elementos=30):
    """ Generar un parallel coordinates plot"""
    df = data.copy()
    df['Clase'] = LabelEncoder().fit_transform(df['Clase'])
    
    plt.figsize=(20, 15)
    plt.suptitle(title, fontsize=16)
    fig = px.parallel_coordinates(df[0: n_elementos], color="Clase",
                                dimensions=variables,
                                color_continuous_scale=px.colors.diverging.Spectral_r,
                                color_continuous_midpoint=2,)
    fig.show()


def contabilizar_registros_duplicados(df):
    """ Metodo para contar los valores duplicados de un dataset"""
    return df.duplicated().sum() 

def contabilizar_valores_nulos(df):
    """ Metodo para contabilizar el numero total de nulos de cada variable del dataset"""
    n_nulls = df.isnull().sum()
    per_nulls = df.isnull().sum() * 100 / len(df)
    df_nulos = pd.concat([n_nulls, per_nulls], axis=1)
    df_nulos.rename(columns = {0:'Total', 1:'%'}, inplace = True)
    return df_nulos

def contabilizar_valores_unicos(df):
    """ Contabilizar valores unicos"""
    for c in df.columns:
        print(c,":", len(df[c].unique().tolist()))

def analisis_correlacion(data,mi_cmap='BrBG'):
    df = data.copy()
    df['Clase'] = LabelEncoder().fit_transform(df['Clase'])

    fig, axes = plt.subplots(1,2,  figsize=(20, 7))
    sns.heatmap(ax=axes[0], data=df.corr(),annot=True, cbar=True,  vmin=-1, vmax=1, fmt='.2f', annot_kws={"size": 6}, yticklabels=df.columns,xticklabels=df.columns, cmap=mi_cmap)
    axes[0].set_title('Correlation Heatmap', fontsize=10)

    sns.heatmap(ax=axes[1], data=df.corr()[['Clase']].sort_values(by='Clase', ascending=False), vmin=-1, vmax=1, annot=True, cmap=mi_cmap)
    axes[1].set_title('Features Correlating with Clase' , fontdict={'fontsize':10})


def draw_cm_and_roc_curve(y_preds, y_test,model_name):
    """ Metodo que pinta la matrix de confusion y el area bajo la curva del modelo"""
    fig, axs = plt.subplots(1,2,figsize=(20,5))
    fig.suptitle(model_name,fontsize=16)
    mat = confusion_matrix(y_preds, y_test)
    names = np.unique(y_preds)
    sns.heatmap(ax=axs[0], data=mat, square=True, annot=True, fmt='d', cmap='Greens', cbar=True, xticklabels=names , yticklabels= names)
    axs[0].set_title('Confusion Matrix')
    axs[0].set_ylabel('True Label')
    axs[0].set_xlabel('Predicted Label')

    auc = roc_auc_score(y_test, y_preds)
    fpr_spe, tpr_sens, thresholds = roc_curve(y_test, y_preds)

    plt.title('Roc Curve. Receiver Operating Characteristic')
    plt.axline((0, 0), (1, 1), linewidth=1, linestyle='--', color='r')
    plt.plot(fpr_spe,tpr_sens, 'g',label="AUC="+str(round(auc,6)))
    plt.ylabel('Sensitivity. True Positive Rate')
    plt.xlabel('1 - Specificity. False Positve Rate')
    plt.legend(loc=4)
    plt.show()

    print("METRICS {}\n".format(model_name))
    print("-------------------------------------------------------------------------------------------")
    print("(exactiud) A = (TP + TN) / (TP + TN + FP + FN).  accuracy_score:",accuracy_score(y_test,y_preds))
    print("(precision) P= TP / TP + FP). precision_score", precision_score(y_test,y_preds))
    print("(sensibilidad) R = TP / (TP + FN). recall_score",recall_score(y_test,y_preds))
    print("(f1-score) F1 = F1 = 2 * ((recall * precision)/(recall + precision)). f1_score:",f1_score(y_test,y_preds))
    print("(Area Under Curve) AUC:",auc)


def draw_curva_roc(y_hat,y_pred):
    fpr, tpr, threshold = roc_curve(y_hat, y_pred)
    roc_auc = roc_auc_score(y_hat, y_pred)

    plt.title('Curva ROC')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('Ratio verdaderos positivos')
    plt.xlabel('Ratios falsos positivos')
    plt.show()

   
def  quitar_nan(df):
    """Metodo para eliminar los valores faltantes"""
    return df.dropna()

def imputar_nan_por_la_mediana(df):
    """ Metodo para imputar los valores nan por la mediana"""
    df_new = df.copy()
    for c_name in df.columns:
        if c_name != 'Clase':
            df_new[c_name] =  df_new[c_name].fillna(df[c_name].median())
    return df_new

def imputar_nan_por_la_media(df):
    """ Metodo para imputar los valores nan por la media"""
    df_new = df.copy()
    for c_name in df.columns:
        if c_name != 'Clase':
            df_new[c_name] =  df_new[c_name].fillna(df[c_name].mean())
    return df_new
 
def eliminar_outiliers(data, variables=['Area','Emayor','Emenor','Exc','Vol','Rat','Perim']):
    """ Trimming values out of limits"""
    df = data.copy()
    for var in variables:
        q1 = df[var].quantile(0.25)
        q3 = df[var].quantile(0.75)
        iqr = q3 - q1 
        upper_limit = q3 + (1.5 * iqr)
        lower_limit = q1 - (1.5 * iqr)
        
        df = df.loc[(df[var]>= lower_limit) & (df[var] <= upper_limit)]
    
    before =  len(data)
    after = len(df)
    print('Before removing outilers:' , before)
    print('After removing outilers:' , after)
    print('outilers:', before - after)
    return df

def capping_outliers(data, variables=['Area','Emayor','Emenor','Exc','Vol','Rat','Perim']):
    """ Change the outliers values to upper o lower limit values""" 
    df = data.copy()
    for var in variables:
        q1 = df[var].quantile(0.25)
        q3 = df[var].quantile(0.75)
        iqr = q3 - q1 
        upper_limit = q3 + (1.5 * iqr)
        lower_limit = q1 - (1.5 * iqr)
        
        df.loc[(df[var] > upper_limit) , var] = upper_limit
        df.loc[(df[var] < lower_limit) , var] = lower_limit

    before =  len(data)
    after = len(df)
    print('Before removing outilers:' , before)
    print('After removing outilers:' , after)
    print('outilers:', before - after)
    return df

 
def estandarizar(df, scaler=StandardScaler()):
    """ In statistics, Standardization is the subtraction of the mean and then dividing by its standard deviation.
        Standardisation assumes that the dataset is in Gaussian distribution and measures the variable at different scales, 
        making all the variables equally contribute to the analysis
    """
    X = df.copy()
    X = X.iloc[:,X.columns !='Clase']
    X_scaled = scaler.fit_transform(X)
    df_scaled = pd.DataFrame(X_scaled, columns=['Area','Emayor','Emenor','Exc','Vol','Rat','Perim'])
    return pd.concat([df_scaled,df['Clase']],axis=1)


def normalizar(df):
    """ 
      In Algebra, Normalization is the process of dividing of a vector by its length and it transforms your data into
      a range between 0 and 1.
    """
    normalizer = Normalizer()
    X = df.copy()
    X = X.iloc[:,X.columns !='Clase']
    X_scaled = normalizer.fit_transform(X)
    df_scaled = pd.DataFrame(X_scaled, columns=['Area','Emayor','Emenor','Exc','Vol','Rat','Perim'])
    return pd.concat([df_scaled,df['Clase']],axis=1)


def test_shapiro_wilk_normality_test(data,var):
    """ Normality tests"""
    stat, p = shapiro(data[var])
    if p > 0.05:
        print('%s is probably Gaussian. stat=%.3f, p=%.3f' % (var,stat, p))
    else:
        print('%s probably is not Gaussian. stat=%.3f, p=%.3f' % (var,stat, p))

def test_dagostino_k_squared_normality_test(data,var):
	k2, p_value = normaltest(data[var])
	print('%s estadistico=%.3f, p-value=%.3f' % (var,k2, p_value))

def show_skew_and_kurtosis(df,variables=['Area','Emayor','Emenor','Exc','Vol','Rat','Perim']):
    """ Metodo que calcula la asimetría y la curtosis"""
    skew = df[variables].skew()
    kurtosis = df[variables].kurtosis()
    df_skew_and_kurtosis = pd.concat([skew, kurtosis], axis=1)
    df_skew_and_kurtosis.rename(columns = {0:'skew', 1:'kurtosis'}, inplace = True)
    return df_skew_and_kurtosis 

def transformacion_logaritmica(data):
    """ Log transformation can help to “stretch out” the tail of a right-skewed distribution, making it more symmetric."""
    df = data.copy()
    df['Clase'] = LabelEncoder().fit_transform(df['Clase'])
    X = df.iloc[:,df.columns !='Clase']
    # Transformacion logaritmica
    df_log = np.log(X)
    df_symmetric = pd.concat([df_log,data['Clase']],axis=1)
    return df_symmetric