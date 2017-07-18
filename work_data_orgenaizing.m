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
