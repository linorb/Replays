% % %% Cell registaration games
% % 
% % CAGE = 13
% % MOUSE = 1
% % path_mouse = ['D:\dev\replays\work_data\two_environments\c', num2str(CAGE), 'm', num2str(MOUSE)];
% % load([path_mouse, '\cellRegisteredFixed.mat'])
% % tempa = optimal_cell_to_index_map(:,1:2:end);
% % tempb = optimal_cell_to_index_map(:,2:2:end);
% % optimal_cell_to_index_map = tempa;
% % save([path_mouse, '\cellRegistered_envA'], 'optimal_cell_to_index_map')
% % optimal_cell_to_index_map = tempb;
% % save([path_mouse, '\cellRegistered_envB'], 'optimal_cell_to_index_map')
% % 
% % 
% % %% my mvmt games
% % CAGE = 40
% % MOUSE = 3
% % path_mouse = ['D:\dev\replays\work_data\recall\c', num2str(CAGE), 'm', num2str(MOUSE),'\day3\linear'];
% % load([path_mouse, '\my_mvmt.mat'])
% % 
% % field_names = fieldnames(my_mvmt{1,3});
% % 
% % for mouse_ind=2:7
% %     temp_field = my_mvmt{1,3}.(field_names{mouse_ind});
% %     temp_field = temp_field(2:end);
% %     my_mvmt{1,3}.(field_names{mouse_ind}) = temp_field;
% % end
% % 
% % save([path_mouse, '\my_mvmt1.mat'], 'my_mvmt')
% % 
% %% create elaborated events matrix
% % MOUSE = [6, 3, 6, 4, 3, 0, 4, 4, 1, 1];
% % CAGE = [40, 40, 38, 38, 38, 38, 6, 7, 11, 13];
% % ENV = {'\linear','\linear','\linear','\linear','\linear','\linear',...
% %     '\envA','\envA','\envA','\envA'};
% % WORK_DIR = {'D:\dev\replays\work_data\recall', ...
% %             'D:\dev\replays\work_data\recall', ...
% %             'D:\dev\replays\work_data\recall', ...
% %             'D:\dev\replays\work_data\recall', ...
% %             'D:\dev\replays\work_data\recall', ...
% %             'D:\dev\replays\work_data\recall', ...
% %             'D:\dev\replays\work_data\two_environments',...
% %             'D:\dev\replays\work_data\two_environments',...
% %             'D:\dev\replays\work_data\two_environments',...
% %             'D:\dev\replays\work_data\two_environments'};
% 
% MOUSE = [4, 4, 1, 1];
% CAGE = [6, 7, 11, 13];
% ENV = {'\envB','\envB','\envB','\envB'};
% WORK_DIR = {'D:\dev\replays\work_data\two_environments',...
%             'D:\dev\replays\work_data\two_environments',...
%             'D:\dev\replays\work_data\two_environments',...
%             'D:\dev\replays\work_data\two_environments'};
% 
% filename = 'results.mat';
% number_of_mice = length(MOUSE);
% 
% for mouse_ind=1:number_of_mice
%     mouse_path = [WORK_DIR{mouse_ind}, '\c', num2str(CAGE(mouse_ind)), ...
%         'm', num2str(MOUSE(mouse_ind))];
%     session_dirs = dir([mouse_path, '\d*']);
%     for session_ind = 1:length(session_dirs)
%         full_path = [mouse_path,'\',session_dirs(session_ind).name,...
%             ENV{mouse_ind}];
%         load([full_path, '\', filename]);
%         number_of_frames = cellActivity{1,1}.numFrames;
%         events = cell(1,length(number_of_frames));
%         number_of_neurons = size(cellActivity{1,1}.Events, 2);
%         allEventsMat = [];
%         for trial_ind=1:length(number_of_frames)
%             events{trial_ind} = zeros(number_of_neurons, ...
%                 number_of_frames(trial_ind));
%             for neuron_ind = 1:number_of_neurons
%                 neuron_activity = cellActivity{1,1}.Events{trial_ind, ...
%                     neuron_ind};
%                 number_of_neuron_events = size(neuron_activity, 2);
%                 for event_ind = 1:number_of_neuron_events
%                     % Setting 1 value from the rise time of the event
%                     end_frame = neuron_activity(1, event_ind);
%                     begin_frame = end_frame - neuron_activity(3, event_ind);
%                     if begin_frame>1
%                         events{trial_ind}(neuron_ind, begin_frame:end_frame) = 1;             
%                     end                   
%                 end
%             end
%             allEventsMat = [allEventsMat, events{trial_ind}];
%         end
%         allEventsMat = allEventsMat';
%         save([full_path,'\FixedEventsMat.mat'], 'allEventsMat');
%     end
% end
%             
% %% smooth velocity
% 
% MOUSE = [6, 3, 6, 4, 3, 0, 4, 4, 1, 1, 4, 4, 1, 1];
% CAGE = [40, 40, 38, 38, 38, 38, 6, 7, 11, 13, 6, 7, 11, 13];
% ENV = {'\linear','\linear','\linear','\linear','\linear','\linear',...
%     '\envA','\envA','\envA','\envA', '\envB','\envB','\envB','\envB'};
% WORK_DIR = {'D:\dev\replays\work_data\recall', ...
%             'D:\dev\replays\work_data\recall', ...
%             'D:\dev\replays\work_data\recall', ...
%             'D:\dev\replays\work_data\recall', ...
%             'D:\dev\replays\work_data\recall', ...
%             'D:\dev\replays\work_data\recall', ...
%             'D:\dev\replays\work_data\two_environments',...
%             'D:\dev\replays\work_data\two_environments',...
%             'D:\dev\replays\work_data\two_environments',...
%             'D:\dev\replays\work_data\two_environments',...
%             'D:\dev\replays\work_data\two_environments',...
%             'D:\dev\replays\work_data\two_environments',...
%             'D:\dev\replays\work_data\two_environments',...
%             'D:\dev\replays\work_data\two_environments'};
%         
% filename = 'my_mvmt.mat';
% 
% % smoothing kernal
% sm_mask = ones(1,5);
% sm_mask = sm_mask/sum(sm_mask);
% 
% number_of_mice = length(MOUSE);
% 
% for mouse_ind=11:number_of_mice
%     mouse_path = [WORK_DIR{mouse_ind}, '\c', num2str(CAGE(mouse_ind)), ...
%         'm', num2str(MOUSE(mouse_ind))];
%     session_dirs = dir([mouse_path, '\d*']); 
%     for session_ind = 1:length(session_dirs)
%         full_path = [mouse_path,'\',session_dirs(session_ind).name,...
%             ENV{mouse_ind}];
%         load([full_path, '\', filename]);
%         number_of_trials = length(my_mvmt);
%         
%         for trial_ind=2:number_of_trials
%             field_names = fieldnames(my_mvmt{1,trial_ind});
%             current_velocity = my_mvmt{1, trial_ind}.(field_names{5});
%             smooth_velocity = conv(current_velocity,sm_mask,'same');
%             my_mvmt{1, trial_ind}.velocity = smooth_velocity;
%         end
%         save([full_path,'\my_mvmt_smooth.mat'], 'my_mvmt');
%     end
% end

%% Change bin numbering of the Lshape
MOUSE = [4, 4, 1, 1];
CAGE = [6, 7, 11, 13];
ENV = {'\envB','\envB','\envB','\envB'};
WORK_DIR = {'D:\dev\replays\work_data\two_environments',...
            'D:\dev\replays\work_data\two_environments',...
            'D:\dev\replays\work_data\two_environments',...
            'D:\dev\replays\work_data\two_environments'};
        
filename = 'my_mvmt_smooth.mat';

mapping_index = [1,13;
                 2,12;
                 3,11;
                 4,10;
                 5,9;
                 6,8;
                 8,6;
                 9,5;
                 10,4;
                 11,3;
                 12,2;
                 13,1];
 
N = size(mapping_index,1);

for mouse_ind=1:length(MOUSE)
    mouse_path = [WORK_DIR{mouse_ind}, '\c', num2str(CAGE(mouse_ind)), ...
        'm', num2str(MOUSE(mouse_ind))];
    session_dirs = dir([mouse_path, '\d*']); 
    for session_ind = 1:length(session_dirs)
        full_path = [mouse_path,'\',session_dirs(session_ind).name,...
            ENV{mouse_ind}];
        load([full_path, '\', filename]);
        new_mvmt = my_mvmt;
        % Change the numbering of bins acording to mapping_index
        for j=2:length(my_mvmt) % go over all cells my_mvmt
            x = my_mvmt{j}.binned;
            ind = false(N,length(x));
            y=x;
            for i=1:N
                a = x == mapping_index(i,1);
                ind(i,:) = a;
            end

            for i=1:N
                y(ind(i,:)) = mapping_index(i,2);
            end
            new_mvmt{j}.binned = y;
        end
        my_mvmt = new_mvmt;
        save([full_path,'\my_mvmt_smooth.mat'], 'my_mvmt');
    end
end




