import numpy as np

def aug_freq_low(X, lim_freq, nb_channels, type_r="int"):
    X_transformed = np.zeros(X.shape)
    for i in range(len(X)):  
        img = X[i]     
        crow = int(img.shape[0]/2)
        ccol = int((img.shape[1]/2))
        if (nb_channels==1):
            f = np.fft.fft2(img[:,:,0])
            fshift = np.fft.fftshift(f)
            mask = np.zeros(f.shape)
            mask[crow-lim_freq:crow+lim_freq, ccol-lim_freq:ccol+lim_freq] = 1
            masked_f = fshift*mask
            f_inv = np.fft.ifftshift(masked_f)
            f_inv = np.fft.ifft2(f_inv)
            X_transformed[i] = np.real(np.expand_dims(f_inv, axis=2)) + np.imag(np.expand_dims(f_inv, axis=2))
        if (nb_channels==3):
            for c in range(3):
                f = np.fft.fft2(img[:,:,c])    
                fshift = np.fft.fftshift(f)
                mask = np.zeros(f.shape)
                mask[crow-lim_freq:crow+lim_freq, ccol-lim_freq:ccol+lim_freq] = 1
                masked_f = fshift*mask
                f_inv = np.fft.ifftshift(masked_f)
                f_inv = np.fft.ifft2(f_inv)
                X_transformed[i,:,:,c] = np.real(f_inv) + np.imag(f_inv)
    if (type_r =="int"):
        return(X_transformed.astype('int')) 
    if (type_r == "float"):
        return(X_transformed) 

def aug_freq_high(X, lim_freq, nb_channels, type_r="int"):
    X_transformed = np.zeros(X.shape)
    for i in range(len(X)):  
        img = X[i]     
        crow = int(img.shape[0]/2)
        ccol = int((img.shape[1]/2))
        if (nb_channels==1):
            f = np.fft.fft2(img[:,:,0])
            fshift = np.fft.fftshift(f)
            mask = np.ones(f.shape)
            mask[crow-lim_freq:crow+lim_freq, ccol-lim_freq:ccol+lim_freq] = 0
            masked_f = fshift*mask
            f_inv = np.fft.ifftshift(masked_f)
            f_inv = np.fft.ifft2(f_inv)
            X_transformed[i] = np.real(np.expand_dims(f_inv, axis=2)) + np.imag(np.expand_dims(f_inv, axis=2))     
        if (nb_channels==3):
            for c in range(3):
                f = np.fft.fft2(img[:,:,c])    
                fshift = np.fft.fftshift(f)
                mask = np.ones(f.shape)
                mask[crow-lim_freq:crow+lim_freq, ccol-lim_freq:ccol+lim_freq] = 0
                masked_f = fshift*mask
                f_inv = np.fft.ifftshift(masked_f)
                f_inv = np.fft.ifft2(f_inv)
                X_transformed[i,:,:,c] = np.real(f_inv) + np.imag(f_inv)
    if (type_r =="int"):
        return(X_transformed.astype('int')) 
    if (type_r == "float"):
        return(X_transformed) 
