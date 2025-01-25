from scipy.stats import entropy, norm
import numpy as np
import warnings
import matplotlib.pyplot as plt
import cmcrameri.cm as cmc
def binom_confidence(n: int, c: float) -> float:
    """
    Returns probability p₁ of successes in binomial distribution
    
    Parameters:
    -----------
    n : int
        Number of trials
    c : float
        Confidence level (e.g., 0.99 for 99% confidence)
    """
    p = 0.5 + norm.ppf(1-(1-c)/2)/(2*np.sqrt(n))
    return min(1.0, p)

def binom_free_entropy(n: int, c: float, base: float=2) -> float:
    """
    Returns the free entropy Hf associated with binom_confidence
    
    Parameters:
    -----------
    n : int
        Number of trials
    c : float
        Confidence level
    base : float
        Base for entropy calculation (default=2)
    """
    p = binom_confidence(n, c)
    return 1 - entropy([p, 1-p], base=base)

def analyze_bit_information(da):
    """
    Analyzes the bitwise information content of a DataArray
    
    Parameters:
    -----------
    da : xarray.DataArray
        Input data array
    
    Returns:
    --------
    information : numpy.ndarray
        Array of information content for each bit
    """
    # Data validation
    if not np.isfinite(da.values).all():
        warnings.warn("Data contains non-finite values")
    
    # Ensure 32-bit float
    if da.values.dtype != np.float32:
        da = da.astype(np.float32)
    
    # Get binary representation using big-endian byte order
    binary = da.values.astype('>f4').tobytes()
    binary = np.frombuffer(binary, dtype=np.uint8)
    bit_counts = np.unpackbits(binary).reshape(-1, 32)
    
    # Calculate probabilities and information content
    probabilities = bit_counts.mean(axis=0)
    information = 1 - entropy([probabilities, 1-probabilities], base=2)
    
    # Calculate significance threshold
    n_elements = np.prod(da.shape)
    M = binom_free_entropy(n_elements, 0.99)  # 99% confidence
    
    # Filter insignificant information (mantissa bits only)
    threshold = max(M, 1.5*np.max(information[-4:]))  # Use max of last 4 bits
    insignificant = (information <= threshold) & (np.arange(32) > 9)
    information[insignificant] = 1e-300  # Small positive number for log scale
    
    return information


def plot_bit_information(ds):
    """
    Creates visualization of bitwise information content for an xarray Dataset
    """
    # Calculate bit information for each variable
    bit_info = {}
    nvars = len(ds.data_vars)
    
    # Calculate and filter information content
    ICfilt = np.zeros((nvars, 32))
    for i, var in enumerate(ds.data_vars):
        ic = analyze_bit_information(ds[var])
        
        # Calculate threshold similar to Julia version
        n_elements = np.prod(ds[var].shape)
        p = binom_confidence(n_elements, 0.99)
        M0 = 1 - entropy([p, 1-p], base=2)  # Changed M₀ to M0
        threshold = max(M0, 1.5*np.max(ic[-4:]))
        
        # Filter insignificant bits (mantissa only)
        insignificant = (ic <= threshold) & (np.arange(32) > 9)
        ic[insignificant] = np.finfo(float).tiny
        ICfilt[i, :] = ic
    
    # Create figure
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.invert_yaxis()
    plt.tight_layout(rect=[0.07, 0.08, 0.95, 0.98])
    pos = ax1.get_position()
    
    # Add colorbar axes
    cax = fig.add_axes([pos.x0, 0.06, pos.x1-pos.x0, 0.02])
    
    # Create twin axes for additional information
    ax1right = ax1.twinx()
    ax1right.invert_yaxis()
    
    # Calculate cumulative information
    ICcsum = np.cumsum(ICfilt, axis=1)
    
    # Calculate 99% and 100% information thresholds
    infbits = np.zeros(nvars)
    infbits100 = np.zeros(nvars)
    for i in range(nvars):
        total_info = ICcsum[i, -1]
        infbits[i] = np.where(ICcsum[i, :] >= 0.99 * total_info)[0][0]
        infbits100[i] = np.where(ICcsum[i, :] >= total_info - np.finfo(float).eps)[0][0]
    
    # Plot heatmap with masked data for better visualization
    import cmcrameri.cm as cmc
    ICnan = np.where(ICfilt > np.finfo(float).tiny, ICfilt, np.nan)
    print("ICfilt unique values:", np.unique(ICfilt[ICfilt > np.finfo(float).tiny]))
    pcm = ax1.pcolormesh(ICnan, vmin=0, vmax=1, cmap=cmc.turku_r)
    cbar = fig.colorbar(pcm, cax=cax, orientation='horizontal')
    cbar.set_label('real information content [bit]')
    
    # Plot 99% information line
    ax1.plot(infbits, np.arange(nvars), 'C1', drawstyle='steps-pre', 
             zorder=10, label='99% of\ninformation')
    
    # Add shading for last 1% information and unused bits
    for i in range(nvars):
        # Grey shading for unused bits (first)

        # Cyan shading for last 1% information (second)
        # ax1.fill_betweenx([i, i+1], 
        #                  [infbits[i], infbits[i]], 
        #                  [infbits100[i], infbits100[i]], 
        #                  alpha=0.1, color='c')
        ax1.fill_betweenx([i, i+1], [infbits100[i], infbits100[i]], [32, 32], 
                         alpha=0.4, color='grey')

        ax1.fill_betweenx([i, i+1], 
                         [infbits[i], infbits[i]], 
                         [infbits100[i], infbits100[i]], 
                         alpha=0.3, facecolor='none', edgecolor='c')
    
    # Set axis limits and labels
    ax1.set_xlim(0, 32)
    ax1.set_ylim(nvars, 0)
    ax1right.set_ylim(nvars, 0)
    
    # Add bit position markers
    ax1.set_xticks([1, 9])
    ax1.axvline(1, color='k', lw=1, zorder=3)
    ax1.axvline(9, color='k', lw=1, zorder=3)
    
    # Add bit type labels
    ax1.text(0, nvars+0.5, "sign", rotation=90)
    ax1.text(2, nvars+0.5, "exponent", rotation=90)
    ax1.text(10, nvars+0.5, "mantissa", rotation=90)
    
    # Set y-axis labels
    ax1.set_yticks(np.arange(nvars))
    ax1right.set_yticks(np.arange(nvars))
    ax1.set_yticklabels(list(ds.data_vars))
    
    # Calculate and display total information per value
    total_info = np.sum(ICfilt, axis=1)
    ax1right.set_yticklabels([f"{x:4.1f}" for x in total_info])
    ax1right.set_ylabel("total information per value [bit]")
    
    # Add bit position numbers
    for i in range(8):
        ax1.text(i+0.5, nvars+0.2, str(i+1), ha='center', va='bottom', 
                fontsize=7, color='darkslategrey')
    for i in range(23):
        ax1.text(i+9.5, nvars+0.2, str(i+1), ha='center', va='bottom', 
                fontsize=7)
    
    # Add legend
    legend_elements = [
        plt.Rectangle((0,0), 1, 1, fc='none', ec='C1', label='99% of information'),
        plt.Rectangle((0,0), 1, 1, fc='c', alpha=0.3, label='last 1% of information'),
        plt.Rectangle((0,0), 1, 1, fc='grey', alpha=0.4, label='unused bits')
    ]
    ax1.legend(handles=legend_elements, loc='upper right', framealpha=0.6)
    
    return fig, ax1, ax1right
def plot_bit_information(ds):
    """
    Creates visualization of bitwise information content for an xarray Dataset
    """
    # Calculate bit information for each variable
    bit_info = {}
    nvars = len(ds.data_vars)
    
    # Calculate and filter information content
    ICfilt = np.zeros((nvars, 32))
    from tqdm import tqdm
    for i, var in tqdm(enumerate(ds.data_vars), total=nvars, desc="Analyzing bit information"):
        ic = analyze_bit_information(ds[var])
        
        # Calculate threshold similar to Julia version
        n_elements = np.prod(ds[var].shape)
        p = binom_confidence(n_elements, 0.99)
        M0 = 1 - entropy([p, 1-p], base=2)  # Changed M₀ to M0
        threshold = max(M0, 1.5*np.max(ic[-4:]))
        
        # Filter insignificant bits (mantissa only)
        insignificant = (ic <= threshold) & (np.arange(32) > 9)
        ic[insignificant] = np.finfo(float).tiny
        ICfilt[i, :] = ic
    
    # Create figure
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.invert_yaxis()
    plt.tight_layout(rect=[0.07, 0.08, 0.95, 0.98])
    pos = ax1.get_position()
    
    # Add colorbar axes
    cax = fig.add_axes([pos.x0, 0.06, pos.x1-pos.x0, 0.02])
    
    # Create twin axes for additional information
    ax1right = ax1.twinx()
    ax1right.invert_yaxis()
    
    # Calculate cumulative information
    ICcsum = np.cumsum(ICfilt, axis=1)
    
    # Calculate 99% and 100% information thresholds
    infbits = np.zeros(nvars)
    infbits100 = np.zeros(nvars)
    for i in range(nvars):
        total_info = ICcsum[i, -1]
        infbits[i] = np.where(ICcsum[i, :] >= 0.99 * total_info)[0][0]
        infbits100[i] = np.where(ICcsum[i, :] >= total_info - np.finfo(float).eps)[0][0]
    
    # Plot heatmap with masked data for better visualization
    import cmcrameri.cm as cmc
    ICnan = np.where(ICfilt > np.finfo(float).tiny, ICfilt, np.nan)
    print("ICfilt unique values:", np.unique(ICfilt[ICfilt > np.finfo(float).tiny]))
    pcm = ax1.pcolormesh(ICnan, vmin=0, vmax=1, cmap=cmc.devon_r) #cmc.devon_r
    cbar = fig.colorbar(pcm, cax=cax, orientation='horizontal')
    cbar.set_label('real information content [bit]')
    
    # Plot 99% information line
    ax1.plot(infbits, np.arange(nvars), 'C1', drawstyle='steps-pre', 
             zorder=10, label='99% of\ninformation')
    
    # Add shading for last 1% information and unused bits
    for i in range(nvars):
        # Grey shading for unused bits (first)
        ax1.fill_betweenx([i, i+1], [infbits100[i], infbits100[i]], [32, 32], 
                         alpha=0.4, color='grey')
        
        # Cyan shading for last 1% information (second)
        ax1.fill_betweenx([i, i+1], 
                         [infbits[i], infbits[i]], 
                         [infbits100[i], infbits100[i]], 
                         alpha=0.3, facecolor='none', edgecolor='c')
    
    # Set axis limits and labels
    ax1.set_xlim(0, 32)
    ax1.set_ylim(nvars, 0)
    ax1right.set_ylim(nvars, 0)
    
    # Add bit position markers
    ax1.set_xticks([1, 9])
    ax1.axvline(1, color='b', lw=1, zorder=10, ls='--')
    ax1.axvline(9, color='b', lw=1, zorder=10, ls='--')
    # Add bit type labels (adjusted positions)
    ax1.text(0.5, nvars+0.9, "sign", rotation=90, fontsize=8)
    ax1.text(4.5, nvars+0.9, "exponent", fontsize=8)
    ax1.text(12, nvars+0.9, "mantissa",  fontsize=8)
    
    # Set y-axis labels
    ax1.set_yticks(np.arange(nvars))
    ax1right.set_yticks(np.arange(nvars))
    ax1.set_yticklabels(list(ds.data_vars))
    
    # Calculate and display total information per value
    total_info = np.sum(ICfilt, axis=1)
    ax1right.set_yticklabels([f"{x:4.1f}" for x in total_info])
    ax1right.set_ylabel("total information per value [bit]")
    
    # Add bit position numbers (adjusted positions)
    for i in range(8):
        ax1.text(i+0.5, nvars+0.7, str(i+1), ha='center', va='bottom', 
                fontsize=7, color='darkslategrey')
    for i in range(23):
        ax1.text(i+9.5, nvars+0.7, str(i+1), ha='center', va='bottom', 
                fontsize=7)
    
    # Add mantissa bits text
    ax1.text(infbits[0]+0.1, 0.8, f"{int(infbits[0]-9)} mantissa bits", 
             fontsize=8, color="saddlebrown")
    for i in range(1, nvars):
        ax1.text(infbits[i]+0.1, i+0.8, f"{int(infbits[i]-9)}", 
                fontsize=8, color="saddlebrown")
    
    # Add legend
    legend_elements = [
        plt.Rectangle((0,0), 1, 1, fc='none', ec='C1', label='99% of information'),
        plt.Rectangle((0,0), 1, 1, fc='c', alpha=0.3, label='last 1% of information'),
        plt.Rectangle((0,0), 1, 1, fc='grey', alpha=0.4, label='unused bits')
    ]
    ax1.legend(handles=legend_elements, loc='upper right', framealpha=0.6)
    
    return fig, ax1, ax1right
