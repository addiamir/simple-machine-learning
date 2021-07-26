### Making subset data for initial plotting
mathpy <- subset(DAXdata, select = c(Math, Pysics))
mathsci <- subset(DAXdata, select = c(Math, Science))
mathstat <- subset(DAXdata, select = c(Math, Statistics))

hist(DAXdata$Math)
hist(DAXdata$Pysics)
hist(DAXdata$Science)
hist(DAXdata$Statistics)




#Setting Graph Presentation Layout
par(mfrow = c(1,3))
#Multiple Plot
boxplot(mathpy, 
        xlim = c(0, 3),
        breaks = 9,
        main = "Math to Pysics",
        ylab = "Grades",
        xlab = " ",
        col = c("red","Purple"))

boxplot(mathsci, 
        xlim = c(0, 3),
        breaks = 9,
        main = "Math to Science",
        ylab = "Grades",
        xlab = " ",
        col = c("red","cyan"))

boxplot(mathstat, 
        xlim = c(0, 3),
        breaks = 9,
        main = "Math to Statistics",
        ylab = "Grades",
        xlab = " ",
        col = c("red","yellow"))