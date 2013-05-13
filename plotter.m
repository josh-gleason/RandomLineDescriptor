function plotter(filename, plotTitle, outputName)
   A = load(filename);
   plot(A(:,1),A(:,2), '-*b','LineWidth',2);
   xlabel('1-precision')
   ylabel('recall')
   title(plotTitle)
   axis([0 1 0 1])
   print(outputName,'-dpdf');
end
