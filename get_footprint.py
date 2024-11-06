from statistics import mean
import numpy as np
import scipy

def get_amps(unit):
    '''Given a 2d array, find the largest amplitude signal from trough to peak for each array element in axis = 0.
    
    unit = mean waveform data of shape 384 x 82'''
    amps = []
    
    for chan in unit:
        trough_idx = np.where(chan==np.min(chan))[0][0]
        peak_idx = np.where(chan[trough_idx:]==chan[trough_idx:].max())[0][0]+trough_idx

        trough_h = chan[trough_idx]
        peak_h = chan[peak_idx]
        amp = (peak_h+abs(trough_h))
        amps.append(amp)
        
    return np.array(amps)

def find_intercept(p,xlim,ylim,deg):
    '''Pretty clunky way of finding the edge intercept coordinate of a straight line from the max amplitude channel, given a desired input angle.
    p = tuple of max amp channel (x,y) in pixel space
    xlim = x limit of the probe in pixel space (7)
    ylim = y limit of the probe in pixel space (47)
    deg = defined angle'''
    if deg==0:
        m = 0
        b = p[1] - m*p[0] 
        x = xlim[1]
        y = p[1]
        
    elif 0<deg<90:
        m = np.tan(np.deg2rad(deg))
        b = p[1] - m*p[0]
        
        dx = abs(p[0] - xlim[1])
        dy = abs(p[1] - ylim[1])
        
        if dx<dy:
            x = xlim[1]
            y = m*x+b
            
            if y>ylim[1]:
                y = ylim[1]
                x = (y-b)/m
        else:
            y = ylim[1]
            x = (y-b)/m
            if x>xlim[1]:
                x = xlim[1]
                y = m*x+b
            
    elif deg == 90:
        m = 0
        b = p[1] - m*p[0] 
        x = p[0]
        y = ylim[1]
    
    elif 90<deg<180:
        m = np.tan(np.deg2rad(deg))
        b = p[1] - m*p[0]
        
        dx = abs(p[0] - xlim[0])
        dy = abs(p[1] - ylim[1])
        
        if dx<dy:
            x = xlim[0]
            y = m*x+b
            
            if y>ylim[1]:
                y = ylim[1]
                x = (y-b)/m
        else:
            y = ylim[1]
            x = (y-b)/m
            if x<xlim[0]:
                x = xlim[0]
                y = m*x+b
        
    elif deg == 180:
        m = 0
        b = p[1] - m*p[0] 
        x = xlim[0]
        y = p[1]
        
    elif 180<deg<270:
        m = np.tan(np.deg2rad(deg))
        b = p[1] - m*p[0]
        
        dx = abs(p[0] - xlim[0])
        dy = abs(p[1] - ylim[0])
        
        if dx<dy:
            x = xlim[0]
            y = m*x+b
            
            if y<ylim[0]:
                y = ylim[0]
                x = (y-b)/m
        else:
            y = ylim[0]
            x = (y-b)/m
            if x<xlim[0]:
                x = xlim[0]
                y = m*x+b
                
    elif deg == 270:
        m = 0
        b = p[1] - m*p[0] 
        x = p[0]
        y = ylim[0]
        
    elif 270<deg<360:
        m = np.tan(np.deg2rad(deg))
        b = p[1] - m*p[0]
        
        dx = abs(p[0] - xlim[1])
        dy = abs(p[1] - ylim[0])
        
        if dx<dy:
            x = xlim[1]
            y = m*x+b
            
            if y<ylim[0]:
                y = ylim[0]
                x = (y-b)/m
        else:
            y = ylim[0]
            x = (y-b)/m
            if x>xlim[1]:
                x = xlim[1]
                y = m*x+b
                
    elif deg == 360:
        m = 0
        b = p[1] - m*p[0]
        x = xlim[1]
        y = p[1]

    
    return x,y

def tolerant_mean(arrs):

    '''Allows for averaging across axis=1 in a 2D ragged array. Averaging does not include arrays with len < current idx'''

    lens = [len(i) for i in arrs]
    arr = np.ma.empty((np.max(lens),len(arrs)))
    arr.mask = True
    for idx, l in enumerate(arrs):
        arr[:len(l),idx] = l
    return arr.mean(axis = -1), arr.std(axis=-1)

def get_footprint_radius(unit, threshold=30,scale=6,unit_shape=(384,82),n_rows=48,n_col=8):

    ''' Given a unit waveform of shape 384 channels x n time samples, return distance metric calculated as the vector length from the maximum amplitude channel about which
    the average amplitude is less than or equal to threshold.
    
    unit = matrix of 384 channels x t samples. Is reshaped into an NP Ultra configuration for distance purposes.
    threshold = arbitrary voltage threshold used to find the footprint.'''
    if unit_shape[0]//n_col==n_rows:

        # get the maximum amplitude per channel. 
        amps = get_amps(unit).reshape(n_rows,n_col)

        # subtract noise by averaging the amplitudes from the 10 lowest amplitude channels and subtracting from all channels.
        amps = amps-np.mean(np.sort(amps.reshape(n_rows*n_col))[:10])

    # work-around for oddly shaped mean unit waveform arrays
    else:
        if unit_shape[0]%n_col==0:
                amps = get_amps(unit).reshape(unit_shape[0]//n_col,n_col)
                amps = amps-np.mean(np.sort(amps.reshape((unit_shape[0]//n_col)*n_col))[:10])
        elif unit_shape[0]%n_col==1:
            unit = unit[1:,:]
            amps = get_amps(unit).reshape(unit.shape[0]//n_col,n_col)
            amps = amps-np.mean(np.sort(amps.reshape((unit.shape[0]//n_col)*n_col))[:10])

    # get channel index for the max. amp. channel in both unwrapped and grid coords.
    max_chan = np.where(amps == np.max(amps))

    row,col = max_chan[1][0],max_chan[0][0]

    vectors = []

    for deg in np.arange(0,360):
        x,y = find_intercept((row,col),(0,(amps.shape[1])-1),(0,(amps.shape[0])-1),deg)
        
        hyp = int(round(np.sqrt((x-row)**2 + (y-col)**2)*scale))

        c,d = np.linspace(row, x, int(hyp)), np.linspace(col, y, int(hyp))

        zi = scipy.ndimage.map_coordinates(amps, np.vstack((d,c)))
        vectors.append(zi)
    
    mean_vector =  tolerant_mean(vectors)[0]
    try:
        footprint = np.where(mean_vector<=threshold)[0][0]
    except:
        footprint=scale  
    if footprint<scale:
        footprint=scale    
    return footprint