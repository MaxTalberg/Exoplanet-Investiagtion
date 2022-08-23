import matplotlib.pyplot as plt
import numpy as np
from dictionary import exo_Dict, star_Dict
import pandas as pd
from IPython.display import Image

# Astronomical constansts
E_r = 6371e3
S_r = 6.96e8
AU = 1.49e11

# Star properties
R_star = star_Dict['solar_radii']*(S_r/E_r)

# Exoplanet properties
R_A, SM_A = exo_Dict['A']['earth_radii'], exo_Dict['A']['semimajor']*(AU/E_r)
R_B, SM_B = exo_Dict['B']['earth_radii'], exo_Dict['B']['semimajor']*(AU/E_r)
R_C, SM_C = exo_Dict['C']['earth_radii'], exo_Dict['C']['semimajor']*(AU/E_r)
R_D, SM_D = exo_Dict['D']['earth_radii'], exo_Dict['D']['semimajor']*(AU/E_r)

#Habitable zone properties
HZ_upper = 1.4862159583679637 * (AU/E_r)
HZ_lower = 0.7961401947919267 * (AU/E_r)

# #Import nasa dataset
nasa = pd.read_csv('nasa_exoplanet_dataset.csv', delimiter=',', header=19)


class visuals():
    
    'This is a class, where all the relevent visualisations can be performed'
    
    def __init__():
        
        return
    
    def plot_system():
        """
        3.1
        Function plots habitable zone visual 
        Uses patches and plt.Circle
        """
        
        print('\033[1mFigure 5:\033[0m' + 'Visual plot of the habitable zone relative to host star and exoplanets.')
        
        plt.figure(figsize=(5,5))
        star = plt.Circle((0, 0), 10*R_star, color='black', label = 'Star E')

        # Exoplanets
        exoplanet_A = plt.Circle((SM_A*0.707, SM_A*0.707), 50*(R_A), color = 'b', label = 'A', zorder = 3)
        exoplanet_B = plt.Circle((SM_B*0.707, SM_B*0.707), 50*(R_B), color = 'orange', label = 'B', zorder = 3)
        exoplanet_C = plt.Circle((SM_C*0.707, SM_C*0.707), 50*(R_C), color = 'green', label = 'C', zorder = 3)
        exoplanet_D = plt.Circle((SM_D*0.707, SM_D*0.707), 50*(R_D), color = 'r', label = 'D', zorder = 3)

        # Orbits
        orbit_A = plt.Circle((0, 0), SM_A, color='black', fill=False)
        orbit_B = plt.Circle((0, 0), SM_B, color='black', fill=False)
        orbit_C = plt.Circle((0, 0), SM_C, color='black', fill=False)
        orbit_D = plt.Circle((0, 0), SM_D, color='black', fill=False)

        # Habitable zone
        hz_upper = plt.Circle((0, 0), HZ_upper, color='black', fill=False, ls = 'dashed')
        hz_lower = plt.Circle((0, 0), HZ_lower, color='black', fill=False, ls = 'dashed')
    
        ax = plt.gca()
        ax.cla()
    
        # limits
        ax.set_xlim((-R_star, 36000))
        ax.set_ylim((-R_star, 36000))
     
        # labels
        #label = ax.annotate("star", xy=(3, 3), fontsize=12)
        #label = ax.annotate("A", xy=(1200, 1400), fontsize=10)
        #label = ax.annotate("B", xy=(1800, 2200), fontsize=10)
        #label = ax.annotate("C", xy=(2600, 3000), fontsize=10)
        #label = ax.annotate("D", xy=(3200, 3700), fontsize=10)
        label = ax.annotate('Habitable Zone', xy=(11000, 19000), fontsize=12)
    
    
        ax.add_patch(star)
        ax.add_patch(exoplanet_A)
        ax.add_patch(exoplanet_B)
        ax.add_patch(exoplanet_C)
        ax.add_patch(exoplanet_D)
        ax.add_patch(orbit_A)
        ax.add_patch(orbit_B)
        ax.add_patch(orbit_C)
        ax.add_patch(orbit_D)
        ax.add_patch(hz_upper)
        ax.add_patch(hz_lower)
        plt.legend(loc="upper right")
        plt.xlabel('Distance (Earth Radius)')
        plt.ylabel('Distance (Earth Radius)')
        
        return
    
    def plot_comparisons():
        """
        Function subplots comparions to exoplanet population
        2 x 1 plot
        Earth Radius v Earth Mass
        Semi Major Axis v Earth Mass
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,5))
        fig.suptitle('Comparisons to Exoplanet Dataset')
        ax1.scatter(nasa['pl_rade'], nasa['pl_masse'], c = 'black', s = 2)
        ax1.scatter(exo_Dict['A']['earth_radii'], exo_Dict['A']['earth_mass'], label = 'A')
        ax1.scatter(exo_Dict['B']['earth_radii'], exo_Dict['B']['earth_mass'], label = 'B')
        ax1.scatter(exo_Dict['C']['earth_radii'], exo_Dict['C']['earth_mass'], label = 'C')
        ax1.scatter(exo_Dict['D']['earth_radii'], exo_Dict['D']['earth_mass'], label = 'D')
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.legend()
        ax1.set_title('Density')
        ax1.set(xlabel='Radius [Earth Radius]', ylabel='Mass [Earth Mass]')
        
        ax2.scatter(nasa['pl_orbsmax'], nasa['pl_masse'], c = 'black', s = 2)
        ax2.scatter(exo_Dict['A']['semimajor'], exo_Dict['A']['earth_mass'], label = 'A')
        ax2.scatter(exo_Dict['B']['semimajor'], exo_Dict['B']['earth_mass'], label = 'B')
        ax2.scatter(exo_Dict['C']['semimajor'], exo_Dict['C']['earth_mass'], label = 'C')
        ax2.scatter(exo_Dict['D']['semimajor'], exo_Dict['D']['earth_mass'], label = 'D')
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.legend()
        ax2.set_title('Formation')
        ax2.set(xlabel='Semi-Major Axis [Au]', ylabel= 'Mass [Earth mass]')
        
        return
    
    def comparisons_image():
        """
        Image of function above
        With added drawings
        """
        return Image(url= "comparisons.jpg", width=900, height=900)
               
        
