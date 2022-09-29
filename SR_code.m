clear;
clc;
addpath('/media/D/alex/spams-matlab-v2.6/test_release');
addpath('/media/D/alex/spams-matlab-v2.6/src_release');
addpath('/media/D/alex/spams-matlab-v2.6/build');
setenv('MKL_NUM_THREADS','1')
setenv('MKL_SERIAL','YES')
setenv('MKL_DYNAMIC','NO')
num=785;
addpath('/media/D/alex/alex_code/NIFTI_20110310');
mask_path='/media/D/alex/GA-4DCNN_project/Others_code/MNI152_4mm_brain_mask_forHCP.nii';
temp_mask=load_untouch_nii(mask_path);
mask=temp_mask.img;
ind = find(mask);
TASK={'emotion','motor','gambling','language','relation','social','WM'};
subject_list=importdata('/media/D/alex/MIA_MH_GAGNN/4DfMRI_input/data_selected_from_HCP900/selected_list.txt');

for i=1:num
    for task_id=1:7
        tic
        temp_path=['/media/D/alex/MIA_MH_GAGNN/4DfMRI_input/data_selected_from_HCP900/',num2str(subject_list(i,1)),'/',TASK{task_id},'/signal.mat'];
        temp_data = importdata(temp_path);
        norm=(temp_data-mean2(temp_data))/std2(temp_data);
        param.K=400;
        param.lambda=0.3;
        param.numThreads=-1;
        param.batchsize=800;
        param.iter=500;
        param.pos=true;
        param.posAlpha=true;
        D=mexTrainDL(norm,param);
        param.pos=true;
        param.posAlpha=true;
        alpha=mexLasso(norm,D,param);
        %save
        newfile = ['/media/D/alex/MIA_MH_GAGNN/4DfMRI_input/dl_result/',num2str(subject_list(i,1)),'/',TASK{task_id}];
        mkdir(newfile);
        temp_path1=['/media/D/alex/MIA_MH_GAGNN/4DfMRI_input/dl_result/',num2str(subject_list(i,1)),'/',TASK{task_id},'/D.mat'];
        temp_path2=['/media/D/alex/MIA_MH_GAGNN/4DfMRI_input/dl_result/',num2str(subject_list(i,1)),'/',TASK{task_id},'/alpha.mat'];
        save(temp_path1,'D');
        save(temp_path2,'alpha');
        toc
    end
end

