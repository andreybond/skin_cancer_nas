dirs = (/home/andrey/works/rtu/skin_cancer_nas/data/interim/_melanoma/processed_data/positive/715 /home/andrey/works/rtu/skin_cancer_nas/data/interim/_melanoma/processed_data/positive/758 /home/andrey/works/rtu/skin_cancer_nas/data/interim/_melanoma/processed_data/positive/708 /home/andrey/works/rtu/skin_cancer_nas/data/interim/_melanoma/processed_data/positive/697 /home/andrey/works/rtu/skin_cancer_nas/data/interim/_melanoma/processed_data/positive/762 /home/andrey/works/rtu/skin_cancer_nas/data/interim/_melanoma/processed_data/positive/1277 /home/andrey/works/rtu/skin_cancer_nas/data/interim/_melanoma/processed_data/positive/719 /home/andrey/works/rtu/skin_cancer_nas/data/interim/_melanoma/processed_data/positive/2480 /home/andrey/works/rtu/skin_cancer_nas/data/interim/_melanoma/processed_data/positive/736 /home/andrey/works/rtu/skin_cancer_nas/data/interim/_melanoma/processed_data/positive/1281 /home/andrey/works/rtu/skin_cancer_nas/data/interim/_melanoma/processed_data/positive/744 /home/andrey/works/rtu/skin_cancer_nas/data/interim/_melanoma/processed_data/positive/693 /home/andrey/works/rtu/skin_cancer_nas/data/interim/_melanoma/processed_data/positive/739 /home/andrey/works/rtu/skin_cancer_nas/data/interim/_melanoma/processed_data/positive/2703 /home/andrey/works/rtu/skin_cancer_nas/data/interim/_melanoma/processed_data/negative/1450 /home/andrey/works/rtu/skin_cancer_nas/data/interim/_melanoma/processed_data/negative/576)
for i in $dirs; do zip -r "${i%/}.zip" "$i"; done