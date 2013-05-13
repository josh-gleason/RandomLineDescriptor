function plotter(filename, PRTitle, PRName, ROCTitle, ROCName)
   A = load(filename);
   plot(A(:,1),A(:,2), '-*b','LineWidth',8);
   xlabel('1-precision','FontSize',12,'fontWeight','bold')
   ylabel('recall','FontSize',12,'fontWeight','bold')
   title(PRTitle,'FontSize',12,'fontWeight','bold')
   axis([0 1 0 1])
   set(gca,'FontSize',12,'fontWeight','bold');
   print('-dpng','\"-s600,400\"',PRName);

   ROCx = flipud([1;A(:,3);0]);
   ROCy = flipud([1;A(:,2);0]);

   ROCTitle = sprintf('%s (%f)',ROCTitle, trapz(ROCx,ROCy));

   plot(ROCx, ROCy, '-*r','LineWidth',8);
   xlabel('False Positive Rate','FontSize',12,'fontWeight','bold');
   ylabel('True Positive Rate','FontSize',12,'fontWeight','bold');
   title(ROCTitle,'FontSize',12,'fontWeight','bold')
   axis([0 1 0 1])
   set(gca,'FontSize',12,'fontWeight','bold');
   print('-dpng','\"-s600,400\"',ROCName);
end
