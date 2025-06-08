import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import math
import warnings
from scipy.optimize import least_squares, fsolve
import multiprocessing as mp
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state
from tqdm import tqdm

warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

def log_logistic_3pl(params, x):
    """3-parameter logistic function for dose-response curves"""
    ec50, hill_slope, max_growth = params
    return max_growth / (1 + (x / ec50)**hill_slope)

def log_logistic_3pl_solve(y_target, params, x_guess):
    """Solve for x given y in 3-parameter logistic function"""
    return fsolve(lambda x: log_logistic_3pl(params, x) - y_target, x0=x_guess)


class ufpf:
    """UFPF class for reading and processing plate format Excel files"""
    
    def __init__(self, path, timepoint_in_sheetnames=False):
        self.path = path
        self.number_of_plates = None
        self.plate_info = None
        self.OD = None
        self.read_data(timepoint_in_sheetnames)
    
    def read_data(self, timepoint_in_sheetnames=False):
        """Read data from Excel file with plate format"""
        # Get the number of sheets in the excel file
        self.number_of_plates = len(pd.ExcelFile(self.path).sheet_names)
        
        # For loop over each sheet read them into a dataframe
        for sheet_id, sheet_name in enumerate(pd.ExcelFile(self.path).sheet_names):
            # Read as-is
            raw = pd.read_excel(self.path, sheet_name=sheet_name, header=None)

            # Locate 96-well plate formatted entries in the excel sheet
            plate_entries = []
            for ix, row in raw.iterrows():
                if np.all(np.array(row.values[1:13]) == np.arange(1,13)):
                        plate_entries.append(ix)

            # 1. Collect OD values
            od_wide = raw.iloc[plate_entries[0]+1:plate_entries[0]+9, 1:13]
            od_wide['Row'] = np.arange(1,9)
            od_long = od_wide.melt(id_vars=['Row'], var_name='Column', value_name='OD')
            od_long['Plate_ID'] = sheet_id
            well_rows = {1:'A', 2:'B', 3:'C', 4:'D', 5:'E', 6:'F', 7:'G', 8:'H'}
            od_long['Well'] = od_long['Row'].map(well_rows) + od_long['Column'].astype(str)

            # 2. Collect plate info
            infos = dict()
            for block_name, block_id in zip(['Antibiotic', 'Dose', 'Strain'], range(1,4)):
                info_wide = raw.iloc[plate_entries[block_id]+1:plate_entries[block_id]+9, 1:13]
                info_wide['Row'] = np.arange(1,9)
                info_long = info_wide.melt(id_vars=['Row'], var_name='Column', value_name=block_name)
                info_long['Plate_ID'] = sheet_id
                well_rows = {1:'A', 2:'B', 3:'C', 4:'D', 5:'E', 6:'F', 7:'G', 8:'H'}
                info_long['Well'] = info_long['Row'].map(well_rows) + info_long['Column'].astype(str)
                infos[block_name] = info_long
  
            # 3. Merge all plate info into single table
            plate_info = pd.merge(infos['Antibiotic'],infos['Dose'], on=['Plate_ID', 'Well', 'Row', 'Column'])
            plate_info = pd.merge(plate_info, infos['Strain'], on=['Plate_ID', 'Well', 'Row', 'Column'])
            
            # If there is timepoint data provided in sheetnames, append this to strain names
            if timepoint_in_sheetnames:
                plate_info['Strain'] = plate_info['Strain'].astype(str) + '_' + sheet_name.split('_')[-1]
                plate_info.loc[plate_info['Strain'].str.contains('nan'), 'Strain'] = 'nan'

            # 4. Stack all plates into one table
            if self.plate_info is None:
                self.plate_info = plate_info
            else:
                self.plate_info = pd.concat([self.plate_info, plate_info])
            
            if self.OD is None:
                self.OD = od_long
            else:
                self.OD = pd.concat([self.OD, od_long])

class static:
    """Simplified dose-response analysis class"""
    
    @staticmethod
    def residuals(params, x, y):
        """Residuals function for curve fitting"""
        return y - log_logistic_3pl(params, x)
    
    @staticmethod
    def fit_model(x_data, y_data, p0):
        """Fit 3-parameter logistic model using standard least squares"""
        return least_squares(static.residuals, p0, args=(x_data, y_data), 
                           method='dogbox', loss='soft_l1')
    
    @staticmethod
    def ransac_fit_3pl(x_data, y_data, p0, residual_threshold=None, max_trials=100, 
                       min_samples=None, random_state=None):
        """
        RANSAC fitting for 3-parameter logistic model
        
        Args:
            x_data: Independent variable data
            y_data: Dependent variable data  
            p0: Initial parameter guess [ec50, hill_slope, max_growth]
            residual_threshold: Threshold for determining inliers (auto if None)
            max_trials: Maximum RANSAC iterations
            min_samples: Minimum samples for fitting (auto if None)
            random_state: Random seed
        
        Returns:
            dict with 'params', 'inlier_mask', 'n_trials_' keys
        """
        x_data = np.asarray(x_data)
        y_data = np.asarray(y_data)
        n_samples = len(x_data)
        
        if n_samples < 4:
            # Fall back to standard fitting for small datasets
            try:
                result = static.fit_model(x_data, y_data, p0)
                return {
                    'params': result.x,
                    'inlier_mask': np.ones(n_samples, dtype=bool),
                    'n_trials_': 1
                }
            except:
                return None
        
        # Set defaults
        if min_samples is None:
            min_samples = max(4, int(0.6 * n_samples))  # At least 4, up to 60% of data
        if residual_threshold is None:
            residual_threshold = np.std(y_data) * 0.5  # Adaptive threshold
        
        random_state = check_random_state(random_state)
        
        best_inlier_count = 0
        best_params = None
        best_inlier_mask = None
        
        for trial in range(max_trials):
            # Randomly sample minimum number of points
            sample_indices = random_state.choice(n_samples, size=min_samples, replace=False)
            x_sample = x_data[sample_indices]
            y_sample = y_data[sample_indices]
            
            try:
                # Fit model to subset
                result = static.fit_model(x_sample, y_sample, p0)
                
                # Calculate residuals for all data points
                y_pred = log_logistic_3pl(result.x, x_data)
                residuals = np.abs(y_data - y_pred)
                
                # Determine inliers
                inlier_mask = residuals <= residual_threshold
                inlier_count = np.sum(inlier_mask)
                
                # Keep best model (most inliers)
                if inlier_count > best_inlier_count:
                    best_inlier_count = inlier_count
                    best_params = result.x
                    best_inlier_mask = inlier_mask
                    
            except:
                continue
        
        if best_params is None:
            # Fall back to standard fitting
            try:
                result = static.fit_model(x_data, y_data, p0)
                return {
                    'params': result.x,
                    'inlier_mask': np.ones(n_samples, dtype=bool),
                    'n_trials_': max_trials
                }
            except:
                return None
        
        # Refit using all inliers for final model
        try:
            x_inliers = x_data[best_inlier_mask]
            y_inliers = y_data[best_inlier_mask]
            final_result = static.fit_model(x_inliers, y_inliers, best_params)
            best_params = final_result.x
        except:
            pass  # Use RANSAC result if refit fails
        
        return {
            'params': best_params,
            'inlier_mask': best_inlier_mask,
            'n_trials_': trial + 1
        }
    

    

    
    @staticmethod
    def calculate_ic50_mic(df_analysis, combination_row, mic_percentage=0.05, use_ransac=True):
        """Calculate IC50 and MIC for specific combination defined by all columns in combination_row"""
        
        # Filter data based on all columns in combination_row
        filtered_data = df_analysis.copy()
        for col, val in combination_row.items():
            if col in df_analysis.columns:
                filtered_data = filtered_data[filtered_data[col] == val]
        
        # Also include 'cells only' antibiotic data
        if 'Antibiotic' in combination_row:
            cells_only_data = df_analysis[df_analysis['Antibiotic'].str.lower() == 'cells only']
            # Filter cells only data by other columns except Antibiotic
            for col, val in combination_row.items():
                if col != 'Antibiotic' and col in cells_only_data.columns:
                    cells_only_data = cells_only_data[cells_only_data[col] == val]
            filtered_data = pd.concat([filtered_data, cells_only_data])
        
        if filtered_data.empty:
            return static._create_fail_result()
        
        # Handle zero doses for log scale
        nonzero_min = filtered_data['Dose'][filtered_data['Dose'] > 0].min()
        log_min_dose = nonzero_min / 10 if not pd.isna(nonzero_min) else 0.01
        filtered_data = filtered_data.copy()
        filtered_data.loc[filtered_data['Dose'] == 0, 'Dose'] = log_min_dose
        filtered_data.loc[filtered_data['Dose'] < nonzero_min, 'Dose'] = nonzero_min
        
        x_data = filtered_data['Dose'].to_numpy().astype(float)
        y_data = filtered_data['OD_final'].to_numpy().astype(float)
        
        try:
            # Initial parameter guess: [ec50, hill_slope, max_growth]
            max_growth_guess = filtered_data.loc[
                filtered_data['Dose'] == filtered_data['Dose'].min(), 
                'OD_final'].median()
            p0 = [np.exp((np.log(np.min(x_data)) + np.log(np.max(x_data))) / 2), 1, max_growth_guess]
            
            # Fit model using RANSAC for robustness to outliers
            if use_ransac:
                ransac_result = static.ransac_fit_3pl(x_data, y_data, p0, random_state=42)
                if ransac_result is None:
                    return static._create_fail_result("RANSAC fitting failed")
                ic50, hill_coeff, max_growth = ransac_result['params']
            else:
                optimizer = static.fit_model(x_data, y_data, p0)
                ic50, hill_coeff, max_growth = optimizer.x
            
            # Calculate MIC
            params = [ic50, hill_coeff, max_growth]
            mic = log_logistic_3pl_solve(mic_percentage * max_growth, params, ic50)[0]
            
            # Generate fit curve
            x_fit = np.logspace(np.log10(x_data.min()/2), np.log10(x_data.max()*2), 100)
            y_fit = log_logistic_3pl(params, x_fit)
            
            result_dict = {
                'Status': 'PASS',
                'IC50': ic50,
                'MIC': mic,
                'max_growth': max_growth,
                'hill_coeff': hill_coeff,
                'ic50_threshold': log_logistic_3pl(params, ic50),
                'mic_threshold': log_logistic_3pl(params, mic),
                'x_fit': json.dumps(x_fit.tolist()),
                'y_fit': json.dumps(y_fit.tolist()),
                'ransac_used': use_ransac
            }
            
            return result_dict
            
        except Exception as e:
            return static._create_fail_result(str(e))
    
    @staticmethod
    def _create_fail_result(error=""):
        """Create a failure result dictionary"""
        return {
            'Status': 'FAIL',
            'IC50': np.nan,
            'MIC': np.nan,
            'max_growth': np.nan,
            'hill_coeff': np.nan,
            'ic50_threshold': np.nan,
            'mic_threshold': np.nan,
            'x_fit': '[]',
            'y_fit': '[]',
            'ransac_used': False,
            'error': error
        }
    
    @staticmethod
    def _process_combination(args):
        """Helper function for multiprocessing"""
        df_analysis, row = args
        return static.calculate_ic50_mic(df_analysis, row)
    
    @staticmethod
    def apply_phenotyper(df_analysis, valid_combinations, threads=8):
        """Apply phenotyping to all valid strain-antibiotic combinations"""
        df_phenotyped = valid_combinations.copy()
        
        # Prepare arguments for multiprocessing
        args_list = [(df_analysis, row) for _, row in valid_combinations.iterrows()]
        
        # Process with multiprocessing
        with mp.Pool(processes=threads) as pool:
            results = list(tqdm(
                pool.imap(static._process_combination, args_list),
                total=len(valid_combinations),
                desc="Processing combinations"
            ))
        
        # Assign results back to dataframe
        for ix, result in enumerate(results):
            for key, val in result.items():
                if key not in df_phenotyped.columns:
                    df_phenotyped[key] = np.nan
                df_phenotyped.iloc[ix, df_phenotyped.columns.get_loc(key)] = val
        
        return df_phenotyped
    
    @staticmethod
    def cap_growth_features_within_experiment_range(growth_features):
        """Cap IC50/MIC values within experimental range"""
        capped = growth_features.copy()
        capped['insufficient_drug'] = False
        
        for ix, row in capped.iterrows():
            if row['x_fit'] and row['x_fit'] != '[]':
                try:
                    x_fit = json.loads(row['x_fit'])
                    if len(x_fit) > 2:
                        median_fold_change = np.median(np.exp(np.diff(np.log(sorted(x_fit)))))
                        allowed_cap = max(x_fit) * median_fold_change**2
                        
                        if allowed_cap < row['IC50']:
                            capped.loc[ix, ['IC50', 'MIC']] = allowed_cap
                            capped.loc[ix, 'insufficient_drug'] = True
                except:
                    continue
        
        return capped
    
    @staticmethod
    def plot_dose_response_curve_errorbar(df_analysis, strain, antibiotic, strain_colors, ax):
        """Plot dose-response curve with error bars"""
        # Filter data
        filtered_data = df_analysis[
            (df_analysis['Strain'].str.lower() == strain.lower()) &
            (df_analysis['Antibiotic'].str.lower() == antibiotic.lower())
        ]
        
        if filtered_data.empty:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center')
            return
        
        # Group by dose and calculate statistics
        stats = filtered_data.groupby('Dose')['OD_final'].agg(['mean', 'std']).reset_index()
        
        # Handle zero doses for log scale
        non_zero_doses = stats['Dose'] > 0
        if non_zero_doses.any():
            min_nonzero = stats.loc[non_zero_doses, 'Dose'].min()
            stats.loc[~non_zero_doses, 'Dose'] = min_nonzero / 2
        
        # Plot
        color = strain_colors.get(strain, 'gray')
        ax.errorbar(stats['Dose'], stats['mean'], yerr=stats['std'], 
                   fmt='-o', color=color, capsize=3, markerfacecolor='none')
        
        # Format plot
        ax.set_xlabel(f'{antibiotic} (μg/mL)')
        ax.set_ylabel('OD_final')
        ax.set_title(f'{strain}')
        ax.set_xscale('log')
        ax.set_ylim(-0.03, 1.1)
        ax.grid(True, linestyle=':')
        
        # Set x-ticks
        all_doses = np.sort(stats['Dose'].unique())
        ax.set_xticks(all_doses)
        labels = [f'{d:.2f}' if d != min_nonzero/2 else '0' for d in all_doses]
        ax.set_xticklabels(labels, rotation=60)
    
    @staticmethod
    def plot_od_final_for_selected_antibiotic(df_analysis, plot_function, 
                                             selected_antibiotic, strain_colors=None):
        """Plot dose-response curves for all strains for a selected antibiotic"""
        if strain_colors is None:
            strain_colors = {}
        
        # Filter data
        df_filtered = df_analysis[df_analysis['Antibiotic'] == selected_antibiotic]
        unique_strains = list(set(df_filtered['Strain'].unique()) - 
                            {'Media Only', 'Cells Only', 'media only', 'cells only'})
        
        if not unique_strains:
            print(f"No strains found for antibiotic: {selected_antibiotic}")
            return
        
        # Create subplots
        num_columns = 4
        num_rows = math.ceil(len(unique_strains) / num_columns)
        
        fig, axs = plt.subplots(num_rows, num_columns, 
                              figsize=(4 * num_columns, 3 * num_rows), dpi=150)
        
        # Handle single subplot case
        if num_rows == 1 and num_columns == 1:
            axs = [axs]
        elif num_rows == 1 or num_columns == 1:
            axs = axs.flatten()
        else:
            axs = axs.flatten()
        
        # Plot each strain
        for i, strain in enumerate(unique_strains):
            plot_function(df_filtered, strain, selected_antibiotic, strain_colors, axs[i])
        
        # Hide unused subplots
        for i in range(len(unique_strains), len(axs)):
            axs[i].axis('off')
        
        fig.suptitle(f'Antibiotic: {selected_antibiotic}', fontsize=14, y=0.97)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_dose_response_curve_fit(df_analysis, growth_features, strain_colors, 
                                   strain, antibiotic, replicate="", ax=None):
        """Plot dose-response curve with fitted curve and IC50/MIC annotations"""
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
        
        # Get experimental data
        filtered_data = df_analysis[
            ((df_analysis['Antibiotic'] == antibiotic) |
             (df_analysis['Antibiotic'].str.lower() == 'cells only')) &
            (df_analysis['Strain'] == strain)
        ]
        
        if replicate:
            filtered_data = filtered_data[filtered_data['Replicate'] == replicate]
        
        color = strain_colors.get(strain.lower(), 'gray')
        
        # Plot experimental points
        if not filtered_data.empty:
            filtered_data = filtered_data.sort_values(by='Dose')
            ax.semilogx(filtered_data['Dose'], filtered_data['OD_final'], 
                       'o', markersize=6, color=color, markerfacecolor='none', 
                       label=f'Data')
        
        # Get fitted results
        result_mask = ((growth_features['Strain'].str.lower() == strain.lower()) &
                      (growth_features['Antibiotic'].str.lower() == antibiotic.lower()))
        
        if replicate:
            result_mask &= (growth_features['Replicate'] == replicate)
        
        if not result_mask.any():
            ax.text(0.5, 0.5, 'No fit data', transform=ax.transAxes, ha='center')
            return
        
        result = growth_features[result_mask].iloc[0]
        
        # Plot fitted curve
        try:
            x_fit = json.loads(result.get('x_fit', '[]'))
            y_fit = json.loads(result.get('y_fit', '[]'))
            
            if x_fit and y_fit:
                ax.semilogx(x_fit, y_fit, '-', color='black', alpha=0.5, label='Fit')
        except:
            pass
        
        # Format plot
        ax.set_ylim(-0.01, 1.01)
        ax.set_xlabel(f'{antibiotic} (μg/mL)')
        ax.set_ylabel('OD_final')
        
        # Add IC50/MIC annotations
        ic50 = result.get('IC50', np.nan)
        mic = result.get('MIC', np.nan)
        insufficient = result.get('insufficient_drug', False)
        
        dagger = '†' if insufficient else ''
        
        if not np.isnan(ic50):
            ax.text(0.99, 0.9, f'IC50{dagger}: {ic50:.1e}', 
                   transform=ax.transAxes, fontsize=9, color='magenta', 
                   ha='right', fontweight='bold')
        
        if not np.isnan(mic):
            ax.text(0.99, 0.8, f'MIC{dagger}: {mic:.1e}', 
                   transform=ax.transAxes, fontsize=9, color='blue', 
                   ha='right', fontweight='bold')
        
        # Add vertical/horizontal lines for IC50/MIC
        try:
            if not np.isnan(ic50) and not np.isnan(result.get('ic50_threshold', np.nan)):
                ax.axvline(x=ic50, color='magenta', linestyle=':', alpha=0.5)
                ax.axhline(y=result['ic50_threshold'], color='magenta', linestyle=':', alpha=0.5)
            
            if not np.isnan(mic) and not np.isnan(result.get('mic_threshold', np.nan)):
                ax.axvline(x=mic, color='blue', linestyle=':', alpha=0.5)
                ax.axhline(y=result['mic_threshold'], color='blue', linestyle=':', alpha=0.5)
        except:
            pass
        
        # ax.text(0.02, 0.96, f'{strain}', transform=ax.transAxes, fontsize=9, va='top')
        ax.set_title(f'{strain} {int(replicate)}')
        ax.grid(False)

    @staticmethod
    def load_plate_info(file_path):
        """Load plate information from Excel file"""
        # Get all sheet names
        sheet_names = pd.ExcelFile(file_path).sheet_names
        
        # If there are multiple sheets in the excel combine
        if len(sheet_names) > 1:
            dfs = []
            # Iterate through each sheet
            for sheet_name in sheet_names:
                # Extract Plate ID from sheet name
                plate_id = sheet_name.split('-')[0].strip().split(' ')[1].strip()
                # Load the data from the sheet
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                # Add Plate_ID column
                df['Plate_ID'] = int(plate_id)
                dfs.append(df)
            # Concatenate all DataFrames
            df_plate_info = pd.concat(dfs, ignore_index=True)
        else:
            df_plate_info = pd.read_excel(file_path)
            if not (('Plate_ID' in df_plate_info.columns) or ('Plate' in df_plate_info.columns)):
                # Add Plate_ID column
                df_plate_info['Plate_ID'] = 1
            # Column name if only plate be graceful
            if 'Plate' in df_plate_info.columns:
                df_plate_info.rename(columns={'Plate':'Plate_ID'}, inplace=True)

        if 'Drug' in df_plate_info.columns:
            df_plate_info.rename(columns={'Drug':'Antibiotic'}, inplace=True)

        return df_plate_info

    @staticmethod
    def read_biotek_OD_data(path):
        """Read OD data from BioTek format Excel file"""
        df_OD_raw = pd.read_excel(path)
        # Identify how many plates exist in the BioTek output file
        found_plate_index = df_OD_raw[df_OD_raw.iloc[:,0] == 'Results'].index
        
        results = []
        for plate_id, i in enumerate(found_plate_index):
            start_row = i+4
            end_row = i+12
            # Extract the block
            data_block = df_OD_raw.iloc[start_row:end_row + 1, 2:14]

            # Map each entry in the data block to the correct well
            for idx, row in enumerate(data_block.values):
                for jdx, value in enumerate(row):
                    row_letter = chr(65 + idx)  # ASCII 'A'
                    well_position = f"{row_letter}{jdx + 1}"

                    results.append({
                        'Plate_ID': plate_id+1,
                        'Well': well_position,
                        'OD': value,
                        'Row': idx + 1,
                        'Column': jdx + 1
                    })

        return pd.DataFrame(results)

    @staticmethod
    def calc_median_background_all_plates(df, df_plate_info, specific_wells=None, 
                                        excluded_plates=None, plot=False):
        """Calculate the common median background OD value for 'Media Only' wells"""
        
        # Filter out specific conditions
        if excluded_plates is not None:
            exclude_condition = (~df_plate_info['Plate_ID'].isin(excluded_plates))
        else:
            exclude_condition = True
        if specific_wells is not None:
            specific_wells_condition = (df_plate_info['Well'].isin(specific_wells))
        else:
            specific_wells_condition = True

        condition = (
            ((df_plate_info['Strain'].str.lower() == 'media only') |
             (df_plate_info['Antibiotic'].str.lower() == 'media only')) &
             exclude_condition &
            specific_wells_condition
        )
        media_only = df_plate_info[condition]

        # Select OD values corresponding to 'Media Only' wells
        media_only_values = media_only.merge(df, on=['Plate_ID', 'Well'], how='left')['OD']
        
        # Calculate and return the median of these OD values
        median_value = media_only_values.median()
        
        if plot:
            # Plotting the histogram of OD values
            plt.figure(figsize=(12, 1.5), dpi=90)
            plt.hist(media_only_values.dropna(), bins=np.arange(0,1,0.01), color='black')
            plt.title('OD Values for Media Only Wells')
            plt.xlabel('OD Values')
            plt.ylabel('Frequency')
            plt.grid(True)
            plt.show()

        return median_value

    @staticmethod
    def apply_background_correction(df, df_plate_info, plate_id, background_common):
        """Apply background correction based on 'Media Only' wells for a specific plate"""
        
        # Filter the current plate information
        df_plate_current = df_plate_info[df_plate_info['Plate_ID'] == plate_id]
        
        # Filter 'Media Only' wells for the current plate
        media_only_wells = df_plate_current[
            (df_plate_current['Strain'].str.lower() == 'media only') |
            (df_plate_current['Antibiotic'].str.lower() == 'media only')
        ]['Well'].tolist()

        if media_only_wells:
            # Get the median OD for media only wells
            media_values = df[(df['Plate_ID'] == plate_id) & 
                            (df['Well'].isin(media_only_wells))]['OD']
            median_value = media_values.median()
        else:
            # Use predefined common background value if no 'Media Only' wells are present
            median_value = background_common

        # Create a new DataFrame for background corrected values
        df_bc = df.copy()

        # Filter all wells related to the current plate
        all_wells = df_plate_current['Well'].tolist()

        # Apply correction to each well in the current plate
        for well in all_wells:
            well_mask = (df_bc['Plate_ID'] == plate_id) & (df_bc['Well'] == well)
            corrected_values = df_bc.loc[well_mask, 'OD'] - median_value
            # Set any negative values to 10^-6
            corrected_values[corrected_values <= 0] = 10**-6
            df_bc.loc[well_mask, 'OD_final'] = corrected_values

        return df_bc[df_bc['Plate_ID'] == plate_id]

    @staticmethod
    def prep_valid_combinations(df_analysis, multiplex=['Antibiotic', 'Replicate', 'Strain']):
        """Prepare valid combinations for analysis"""
        # Filter out 'media only' strains and select relevant columns
        valid_combinations = df_analysis[
            (df_analysis['Strain'].str.lower() != 'media only') &
            (df_analysis['Strain'].str.lower() != 'cells only') &
            (df_analysis['Antibiotic'].str.lower() != 'media only') &
            (df_analysis['Antibiotic'].str.lower() != 'cells only')
        ][['Antibiotic', 'Replicate', 'Strain', 'Plate_ID']]

        # Drop duplicates based on multiplex columns
        valid_combinations = valid_combinations.drop_duplicates(subset=multiplex)

        # Eliminate rows where any required column doesn't exist
        valid_combinations.dropna(subset=['Antibiotic', 'Replicate', 'Strain', 'Plate_ID'], inplace=True)

        # Reset the index after dropping rows
        valid_combinations.reset_index(drop=True, inplace=True)

        return valid_combinations[multiplex]

    @staticmethod
    def read_OD_data(path, df_plate_info, drop_last_n_cols=4):
        """Read OD data from Excel and convert to long format in one step
        
        Args:
            path: Path to Excel file
            df_plate_info: DataFrame with plate information 
            drop_last_n_cols: Number of columns to drop from the right (default: 4)
        
        Returns:
            DataFrame in long format with columns: Plate_ID, Well, OD, Row, Column
        """
        # Read Excel file and drop last columns
        df_OD_raw = pd.read_excel(path)
        if drop_last_n_cols > 0:
            df_OD_raw = df_OD_raw.iloc[:,:-drop_last_n_cols]
        
        # Get unique plate numbers
        unique_plates = df_plate_info['Plate_ID'].unique()
        plate_numbers = sorted(list(unique_plates))
        
        # Convert to long format
        results = []
        for n in plate_numbers:
            # Calculate row indices for stacked 8x12 blocks
            start_row = 10 * n - 9 - 1  # 0-based indexing
            end_row = 10 * n - 2 - 1
            
            # Extract the 8x12 data block
            data_block = df_OD_raw.iloc[start_row:end_row + 1, 1:13]
            
            # Convert block to long format
            for idx, row in enumerate(data_block.values):
                for jdx, value in enumerate(row):
                    row_letter = chr(65 + idx)  # A, B, C, ...
                    well_position = f"{row_letter}{jdx + 1}"
                    
                    results.append({
                        'Plate_ID': n,
                        'Well': well_position,
                        'OD': value,
                        'Row': idx + 1,
                        'Column': jdx + 1
                    })
        
        return pd.DataFrame(results)

 