%% Cell registaration games

CAGE = 13
MOUSE = 1
path_mouse = ['D:\dev\replays\work_data\two_environments\c', num2str(CAGE), 'm', num2str(MOUSE)];
load([path_mouse, '\cellRegisteredFixed.mat'])
tempa = optimal_cell_to_index_map(:,1:2:end);
tempb = optimal_cell_to_index_map(:,2:2:end);
optimal_cell_to_index_map = tempa;
save([path_mouse, '\cellRegistered_envA'], 'optimal_cell_to_index_map')
optimal_cell_to_index_map = tempb;
save([path_mouse, '\cellRegistered_envB'], 'optimal_cell_to_index_map')


%% my mvmt games
CAGE = 40
MOUSE = 3
path_mouse = ['D:\dev\replays\work_data\recall\c', num2str(CAGE), 'm', num2str(MOUSE),'\day3\linear'];
load([path_mouse, '\my_mvmt.mat'])

field_names = fieldnames(my_mvmt{1,3});

for mouse_ind=2:7
    temp_field = my_mvmt{1,3}.(field_names{mouse_ind});
    temp_field = temp_field(2:end);
    my_mvmt{1,3}.(field_names{mouse_ind}) = temp_field;
end

save([path_mouse, '\my_mvmt1.mat'], 'my_mvmt')

%% create elaborated events matrix
MOUSE = [6, 3, 6, 4, 3, 0, 4, 4, 1, 1];
CAGE = [40, 40, 38, 38, 38, 38, 6, 7, 11, 13];
ENV = {'\linear','\linear','\linear','\linear','\linear','\linear',...
    '\envA','\envA','\envA','\envA'};
WORK_DIR = {'D:\dev\replays\work_data\recall', ...
            'D:\dev\replays\work_data\recall', ...
            'D:\dev\replays\work_data\recall', ...
            'D:\dev\replays\work_data\recall', ...
            'D:\dev\replays\work_data\recall', ...
            'D:\dev\replays\work_data\recall', ...
            'D:\dev\replays\work_data\two_environments',...
            'D:\dev\replays\work_data\two_environments',...
            'D:\dev\replays\work_data\two_environments',...
            'D:\dev\replays\work_data\two_environments'};
        
filename = 'results.mat';
number_of_mice = length(MOUSE);

for mouse_ind=1:number_of_mice
    mouse_path = [WORK_DIR{mouse_ind}, '\c', num2str(CAGE(mouse_ind)), ...
        'm', num2str(MOUSE(mouse_ind))];
    session_dirs = dir([mouse_path, '\d*']);
    for session_ind = 1:length(session_dirs)
        full_path = [mouse_path,'\',session_dirs(session_ind).name,...
            ENV{mouse_ind}];
        load([full_path, '\', filename]);
        number_of_frames = cellActivity{1,1}.numFrames;
        events = cell(1,length(number_of_frames));
        number_of_neurons = size(cellActivity{1,1}.Events, 2);
        allEventsMat = [];
        for trial_ind=1:length(number_of_frames)
            events{trial_ind} = zeros(number_of_neurons, ...
                number_of_frames(trial_ind));
            for neuron_ind = 1:number_of_neurons
                neuron_activity = cellActivity{1,1}.Events{trial_ind, ...
                    neuron_ind};
                number_of_neuron_events = size(neuron_activity, 2);
                for event_ind = 1:number_of_neuron_events
                    % Setting 1 value from the rise time of the event
                    end_frame = neuron_activity(1, event_ind);
                    begin_frame = end_frame - neuron_activity(3, event_ind);
                    if begin_frame>1
                        events{trial_ind}(neuron_ind, begin_frame:end_frame) = 1;             
                    end                   
                end
            end
            allEventsMat = [allEventsMat, events{trial_ind}];
        end
        allEventsMat = allEventsMat';
        save([full_path,'\FixedEventsMat.mat'], 'allEventsMat');
    end
end
            
