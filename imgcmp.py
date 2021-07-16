import cv2
from image_similarity_measures.quality_metrics import rmse, ssim
from matplotlib import pyplot as plt

epochs = input("Number of training epochs: ")

# Byol
# b_saliency = cv2.imread('./CaptumImages/BYOL_' + epochs + 'ep_saliency.png')
# b_intgrad = cv2.imread('./CaptumImages/BYOL_' + epochs + 'ep_intgrad.png')
# Moco
# b_saliency = cv2.imread('m_sal')
# b_intgrad = cv2.imread('m_intgrad')

# SimCLR
sc_saliency = cv2.imread('./CaptumImages/SimCLR_' + epochs + 'ep_saliency.png')
sc_intgrad = cv2.imread('./CaptumImages/SimCLR_' + epochs + 'ep_intgrad.png')

# SimSiam
ss_saliency = cv2.imread('./CaptumImages/SimSiam_' + epochs + 'ep_saliency.png')
ss_intgrad = cv2.imread('./CaptumImages/SimSiam_' + epochs + 'ep_intgrad.png')

# Calculate rmse: Root Mean Square Error
# b_sc_sal_rmse = rmse(b_saliency, sc_saliency)
# b_ss_sal_rmse = rmse(b_saliency, ss_saliency)
sc_ss_sal_rmse = rmse(sc_saliency, ss_saliency)

# b_sc_int_rmse = rmse(b_intgrad, sc_intgrad)
# b_ss_int_rmse = rmse(b_intgrad, ss_intgrad)
sc_ss_int_rmse = rmse(sc_intgrad, ss_intgrad)

# Calculate ssim:
# b_sc_sal_ssim = ssim(b_saliency, sc_saliency)
# b_ss_sal_ssim = ssim(b_saliency, ss_saliency)
sc_ss_sal_ssim = ssim(sc_saliency, ss_saliency)

# b_sc_int_ssim = ssim(b_intgrad, sc_intgrad)
# b_ss_int_ssim = ssim(b_intgrad, ss_intgrad)
sc_ss_int_ssim = ssim(sc_intgrad, ss_intgrad)

#TODO: print images for better comprehension

with open ('./Reports/comparison_' + epochs + 'ep.txt', mode='w') as savefile:
    savestr=('Saliency\n')
    savestr=savestr + "RMSE: SimCLR-SimSiam: " + str(sc_ss_sal_rmse) + "\n"
    #savestr=savestr + "      BYOL-SimSiam:   " + str(b_ss_sal_rmse) + "\n"
    #savestr=savestr + "      BYOL-SimCLR:    " + str(b_sc_sal_rmse) + "\n\n"
    savestr=savestr + "SSIM: SimCLR-SimSiam: " + str(sc_ss_sal_ssim) + "\n"
    #savestr=savestr + "      BYOL-SimSiam:   " + str(b_ss_sal_ssim) + "\n"
    #savestr=savestr + "      BYOL-SimCLR:    " + str(b_sc_sal_ssim) + "\n\n\n"
    savestr=savestr + 'Integrated Gradients' + "\n"
    savestr=savestr + "RMSE: SimCLR-SimSiam: " + str(sc_ss_int_rmse) + "\n"
    #savestr=savestr + "      BYOL-SimSiam:   " + str(b_ss_int_rmse) + "\n"
    #savestr=savestr + "      BYOL-SimCLR: " + str(b_sc_int_rmse) + "\n\n"
    savestr=savestr + "SSIM: SimCLR-SimSiam: " + str(sc_ss_int_ssim) + "\n"
    #savestr=savestr + "      BYOL-SimSiam:   " + str(b_ss_int_ssim) + "\n"
    #savestr=savestr + "      BYOL-SimCLR: " + str(b_sc_int_ssim)
    savefile.write(savestr)

