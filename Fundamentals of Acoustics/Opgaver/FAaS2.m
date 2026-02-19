%FAaS lecture 2
[chris,fs] = audioread("02-chris_&_chokoladefabrikken_#1.mp3");
Chris=timetable(seconds((0:length(chris)-1)'/fs),chris);
[priestess,fs2] = audioread("02.Lay Down.mp3");
Priestess=timetable(seconds((0:length(priestess)-1)'/fs2),priestess);
