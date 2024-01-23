function  plothebidata(bagfoldername, joint)
%
%   plothebidata(bagfoldername, joint)
%
%   Plot the /joint_states and /joint_commands topics saved in the bag
%   folder.  If 'bagfoldername' is not given or given as 'latest', use
%   the most recent bag folder.  If 'joint' is given and not 'all',
%   plot only the joint given by an index (number) or name (string).
%

%
%   Check the arguments
%
% If no bagfile is specified, use the most recent.
if (~exist('bagfoldername') || strcmp(bagfoldername, 'latest'))
    bagfoldername = latestbagfoldername();
end

% If no joint is given, use all joints.
if (~exist('joint'))
    joint = 'all';
end


%
%   Read the data.
%
% Load the bag.
if ~exist(bagfoldername, 'file')
    error(['ROS bag folder ''' bagfoldername ''' does not exist']);
end
try
    % Matlab R2021 assumes an older ROS bag folder naming convention.
    % So we have to adjust the Humble-generated folder to work.
    if all(extractBefore(version('-release'), 5) == '2021')
        bagfoldername = ros2bagR2021fix(bagfoldername);
    end

    % Load using the regular function.  Note this was renamed in
    % R2022b to ros2bagreader(), but ros2bag() still works.
    bag = ros2bag(bagfoldername);
catch ME
    % Let the user know if the bag is incomplete.
    if (strcmp(ME.identifier, 'ros:mlros2:bag:YAMLFileNotFound'))
        disp('Recording not complete...  Was the recording stopped?');
    else
        disp(['Unable to read from the bag folder ''' bagfoldername '''']);
    end
    % Other possible error...
    % elseif (strcmp(ME.identifier, 'shared_robotics:validation:FileNotExist'))

    % But rethrow the error anyway to abort.
    rethrow(ME);
end

% Grab the bag's start time in seconds.  Go back 10ms, as the first
% message may originated one cycle earlier.
t0 = double(bag.StartTime) * 1e-9 - 0.010;

% Grab the /joint_states and /joint_commands messages.
actmsgs = bagmsgs(bag, '/joint_states');
cmdmsgs = bagmsgs(bag, '/joint_commands');

% Proceed depending on what exists.
if (length(actmsgs) > 0)
    if (length(cmdmsgs) > 0)
        % Both exist!
        [ta, pa, va, ea, namesa] = jointstatedata(actmsgs, joint);
        [td, pd, vd, ed, namesd] = jointstatedata(cmdmsgs, joint);

        if ((length(namesa) ~= length(namesd)) || ...
            (~all(cellfun(@strcmp, namesa, namesd))))
            error('Mismatched joints in /joint_states and /joint_commands');
        end
        names = namesa;

        te = ta;
        pe = pa - interp1(td, pd, te);
        ve = va - interp1(td, vd, te);
        ee = ea - interp1(td, ed, te);
    else
        % Only actual messages exist.
        disp('Showing actual data only');
        [ta, pa, va, ea, names] = jointstatedata(actmsgs, joint);
        [td, pd, vd, ed       ] = deal([], [], [], []);
        [te, pe, ve, ee       ] = deal([], [], [], []);
    end
else
    if (length(cmdmsgs) > 0) 
        % Only command messages exist.
        disp('Showing command data only');
        [ta, pa, va, ea       ] = deal([], [], [], []);
        [td, pd, vd, ed, names] = jointstatedata(cmdmsgs, joint);
        [te, pe, ve, ee       ] = deal([], [], [], []);
    else
        % No data.
        error('No data');
    end
end        

% Shift the initial time, to be relative to the bag's start time.
ta = ta - t0;
td = td - t0;
te = te - t0;


%
%   Plot the absolute values on Figure 11.
%
figure(11);
clf;

% Plot.
ax(1) = subplot(3,1,1);
plot(ta, pa, '-', 'LineWidth', 2);
hold on;
set(gca, 'ColorOrderIndex', 1);
plot(td, pd, '--', 'LineWidth', 2);
grid on;
legend(names);
ylabel('Position (rad)');
figtitle = sprintf('%s - actual (solid) and command (dashed)', bagfoldername);
title(figtitle, 'Interpreter', 'none');

ax(2) = subplot(3,1,2);
plot(ta, va, 'Linewidth', 2);
hold on;
set(gca, 'ColorOrderIndex', 1);
plot(td, vd, '--', 'LineWidth', 2);
grid on;
ylabel('Velocity (rad/sec)');

ax(3) = subplot(3,1,3);
plot(ta, ea, 'Linewidth', 2);
hold on;
set(gca, 'ColorOrderIndex', 1);
plot(td, ed, '--', 'LineWidth', 2);
grid on;
ylabel('Torque (Nm)');
xlabel('Time (sec)');

linkaxes(ax, 'x');

% Name the Figure and span the full 8.5x11 page.
set(gcf, 'Name',          'Joint Data');
set(gcf, 'PaperPosition', [0.25 0.25 8.00 10.50]);

% Return to the top subplot, so subsequent title()'s go here...
subplot(3,1,1);
drawnow;


%
%   Plot the errors on Figure 12.
%
if (length(te) == 0)
    return
end

figure(12);
clf;

% Plot.
ax(1) = subplot(3,1,1);
plot(te, pe, 'LineWidth', 2);
grid on;
legend(names);
ylabel('Position (rad)');
figtitle = sprintf('%s - error (actual-command)', bagfoldername);
title(figtitle, 'Interpreter', 'none');

ax(2) = subplot(3,1,2);
plot(te, ve, 'Linewidth', 2);
grid on;
ylabel('Velocity (rad/sec)');

ax(3) = subplot(3,1,3);
plot(te, ee, 'Linewidth', 2);
grid on;
ylabel('Torque (Nm)');
xlabel('Time (sec)');

linkaxes(ax, 'x');

% Name the Figure and span the full 8.5x11 page.
set(gcf, 'Name',          'Joint Errors');
set(gcf, 'PaperPosition', [0.25 0.25 8.00 10.50]);

% Return to the top subplot, so subsequent title()'s go here...
subplot(3,1,1);
drawnow;

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function  bagfoldername = ros2bagR2021fix(bagfoldername)
%
%   bagfoldername = ros2bagR2021fix(bagfoldername)
%
%   For R2021, Matlab assumes a bag folder/file naming
%   convention of:
%       bagname_0/bagname_0.db3
%   but ROS Humble stores (and R2022/following expects)
%   with a convention of:
%       bagname/bagname_0.db3
%
%   So, assuming we are in the latter situation, we rename the folder
%   by appending a '_0', so R2021 can read it.  Note this makes it
%   un-readable to R2022.  Just remove the trailing '_0' in the folder
%   name to undo...
%

% Remove any trailing '/'
if (bagfoldername(end) == '/')
    bagfoldername = bagfoldername(1:end-1);
end

% Pull the foldername into the path and local folder name.
[path, name, ext] = fileparts(bagfoldername);
bagname = strcat(name, ext);

% We're actually good if the ROS bag inside the folder already matches
% the folder name...
if exist(strcat(bagfoldername, '/', bagname, '.db3'), 'file')
    return;
end

% Ok, so we can proceed to try to fix.
newbagfoldername = strcat(bagfoldername, '_0');
disp('Attempting to adjust ROS Humble BAG to Matlab R2021 by');
disp(['renaming folder ''' bagfoldername ''' to ''' newbagfoldername '''']);

% Abort if appending '_0' doesn't lead to a ROS bag.
if ~exist(strcat(bagfoldername, '/', bagname, '_0.db3'), 'file')
    disp('ABORTING: NO _0.db3 ROS BAG IN FOLDER');
    return;
end

% Abort if there are many ROS bags.
if (length(dir(strcat(bagfoldername, '/*.db3'))) > 1)
    disp('ABORTING: Multiple ROS BAGS IN FOLDER');
    return;
end

% Abort if renaming the folder collides with an existing folder.
if exist(newbagfoldername, 'file')
    disp('ABORTING: destination folder already exists');
    return;
end

% Change the folder name to match the bag name.
[status, msg] = movefile(bagfoldername, newbagfoldername);
if (~status)
    disp(['ABORTING: Unable to rename: ' msg]);
    return;
end

% Success.  Update the used name as well.
bagfoldername = newbagfoldername;
disp(['Successfully renamed.  Proceeding with ''' bagfoldername '''']);

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function bagfoldername = latestbagfoldername()
%
%   bagfoldername = latestbagfoldername()
%
%   Return the name of the latest bag folder including a bag file.
%   Error if there are no bag folders.
%

% Get a list of bag files in subfolders of the current folder.
d = dir('*/*.db3');

% Make sure we have at least one bag file.
if (~size(d,1))
    error('Unable to find a bag folder (including a bag file)');
end

% Find the most recently modified bag file.
[~, idx] = max([d.datenum]);

% Determine the folder that holds the bag file.
[root, name, ext] = fileparts(d(idx).folder);
bagfoldername = strcat(name,ext);

% Report.
disp(['Using bag folder ''' bagfoldername '''']);

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function  msgs = bagmsgs(bag, topicname)
%
%   msgs = bagmsgs(bag, topicname)
%
%   Extract the messages of the named topic from the given ROS2 bag.
%   The messages are returned as a struct array.  The structure
%   contains MessageType as well as the fields of the topic.
%

% Isolate the specified topic.
topic = select(bag, 'Topic', topicname);
if (~topic.NumMessages)
    warning(['No messages under topic ''' topicname '''']);
end

% Convert the messages in the topic into structure array.
msgs = cell2mat(readMessages(topic));

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function  [t, pos, vel, eff, names] = jointstatedata(msgs, joint)
%
%   [t, pos, vel, eff, names] = jointstatedata(msgs, joint)
%
%   Extract the data from the given JointStates messages.  Time is
%   absolute.  The return data gives a row per time sample, and a
%   column per joint (NaN is no data is available).
%
%   If 'joint' is given and not 'all', return only the joint given by
%   an index (number) or name (string).
%

% Check the number of messages/samples (M).
M = length(msgs);
if (M == 0)
    error('No messages/samples');
end

% Double-check the type.
if (~strcmp(msgs(1).MessageType, 'sensor_msgs/JointState'))
    error('Messages are not of type sensor_msgs/JointState');
end

% Check the number of joints (N) from the names.
names = msgs(1).name;
N = length(names);
if (N == 0)
    error('Messages contain no named joints');
end

% Extract the absolute time (from sec/nsec), do not subtract the first time.
headers = vertcat(msgs.header);
stamps  = vertcat(headers.stamp);

sec  = double(vertcat(stamps.sec));
nsec = double(vertcat(stamps.nanosec));
t    = sec + 1e-9*nsec;


% Extract the data. If no data, use NaNs. If mismatched, error. If
% good, process whether individual elements are row or column vectors.
Npos = length(msgs(1).position);
Nvel = length(msgs(1).velocity);
Neff = length(msgs(1).effort);

if     (Npos == 0),  pos = NaN(M,N);
elseif (Npos == N),  pos = reshape(horzcat(msgs.position), N, M)';
else
    error('Mismatched number of joints in position and name fields');
end
if     (Nvel == 0),  vel = NaN(M,N);
elseif (Nvel == N),  vel = reshape(horzcat(msgs.velocity), N, M)';
else
    error('Mismatched number of joints in velocity and name fields');
end
if     (Neff == 0),  eff = NaN(M,N);
elseif (Neff == N),  eff = reshape(horzcat(msgs.effort), N, M)';
else
    error('Mismatched number of joints in effort and name fields');
end

% Potentially isolate a single joint.
if (exist('joint') && (~strcmp(joint, 'all')))
    % For a numeric joint specification.
    if (isnumeric(joint))
        if (numel(joint) ~= 1)
            error('Bad joint index specification');
        end

        % Grab the joint number and use as index.
        ind = floor(joint);
        disp(['Using only joint index ' num2str(ind)]);

    % For a joint name specification.
    elseif (ischar(joint))
        % Find the index for the joint name.
        ind = find(strcmp(names, joint));
        if (isempty(ind))
            error(['Unable to isolate joint ''' joint ''''])
        end
        disp(['Using only joint ''' joint '''']);

    % Otherwise can't do anything.
    else
        error('Bad joint argument');
    end

    % Make sure this is a valid index.
    if ((ind < 1) || (ind > N))
        error(['Out of range joint index ' num2str(ind)]);
    end

    % Isolate the data.
    names = names(ind);
    pos   = pos(:,ind);
    vel   = vel(:,ind);
    eff   = eff(:,ind);
end

end
