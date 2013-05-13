function plotter(filename, PRTitle, PRName, ROCTitle, ROCName)
   A = load(filename);
   plot(A(:,1),A(:,2), '-*b','LineWidth',2);
   xlabel('1-precision')
   ylabel('recall')
   title(PRTitle)
   axis([0 1 0 1])
   print(PRName,'-dpdf');

   ROCx = flipud([1;A(:,3)]);
   ROCy = flipud([1;A(:,2)]);

   ROCTitle = sprintf('%s (%f)',ROCTitle, trapz(ROCx,ROCy));

   plot(ROCx, ROCy, '-*r','LineWidth',2);
   xlabel('False Positive Rate');
   ylabel('True Positive Rate');
   title(ROCTitle)
   axis([0 1 0 1])
   print(ROCName,'-dpdf');
end
