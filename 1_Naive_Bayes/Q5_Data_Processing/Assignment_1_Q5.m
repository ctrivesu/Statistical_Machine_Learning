clc;
clear all;
hold on;

%Reading from File
temp = readtable('C:\Users\Sushant\Desktop\ASU Courses\CSE575 - Statistical Machine Learning\Assignment - 1\SML_HW_1\Q5\trial.csv');

%Q1 - Plotting Histogram
figure(1);
histogram(table2array(temp(:,13)), 'FaceColor', 'g')
histogram(table2array(temp(:,15)), 'FaceColor', 'r')
title('Freq Histogram for GapBetweenPACs (green) and ActionsinPAC (red)');
xlabel('Variable');
ylabel('Frequency');


%Q2 - Plotting Scatter figure
figure(2);
scatter(table2array(temp(:,8)), table2array(temp(:,9)));
title('UniqueHotkeys vs AssigntoHotkeys')
xlabel('AssigntoHotkeys')
ylabel('UniqueHotkeys')


figure(3);
scatter(table2array(temp(:,10)), table2array(temp(:,11)));
title('MinimapAttacks vs MinimapRightClicks')
xlabel('MinimapAttacks')
ylabel('MinimapRightClicks')

%Q3 - PCC calculation
PCC = corrcoef(table2array(temp(:,6:20)))

%PCC - Write to file code
fileID = fopen('C:\Users\Sushant\Desktop\ASU Courses\CSE575 - Statistical Machine Learning\Assignment - 1\SML_HW_1\Q5\PCC_stored.txt','w');
formatspec = '%s ';
for i=1:15
    fprintf(fileID, formatspec, string(PCC(i,:)));
    fprintf(fileID, '\n \n');
end
fclose(fileID);

%Min and Max of PCC
PCC_max = max(max(PCC));
PCC_min = min(min(PCC));

[M, max_row] = max(PCC);
[PCC_max, max_col] = max(M);
max_row = max_row(max_col);

[M, min_row] = min(PCC);
[PCC_min, min_col] = min(M);
min_row = min_row(min_col);

disp(PCC(min_row,min_col));

%Scatter for min
figure(4);
scatter(table2array(temp(:,min_row+5)), table2array(temp(:,min_col+5)));
title('NumberOfPACs vs ActionLatency')
xlabel('ActionLatency')
ylabel('NumberofPACs')



%Scatter for max
figure(5);
max_row = 1;
max_col = 2;
scatter(table2array(temp(:,max_row+5)), table2array(temp(:,max_col+5)));
title('APM vs SelectByHotKeys')
xlabel('APM')
ylabel('SelectByHotKeys')

%CODE TO CLOSE GRAPHS
disp('Enter any key to close all graphs');
pause;
close all;






%{
FILE OPEN CODE
fid = fopen('C:\Users\Sushant\Desktop\ASU Courses\CSE575 - Statistical Machine Learning\trial.csv', 'rt');  
C = textscan('%f,%f','HeaderLines',8);
fclose(fid);
%}



%BUTTON PRESS CODE
%f = figure(3);
%{
w = waitforbuttonpress;
if w == 0
    disp('Button click')
else
    disp('Key press')
end
%}

