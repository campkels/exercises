#----------------------------------------------------------------
# RTI CDS Analytics Exercise 01 - Kelsey Campbell - 1/25/2016
#----------------------------------------------------------------

setwd("C:/Users/Kelsey/Google Drive/jobs/exercises/exercise01/RTI")

install.packages("ggplot2")
library(ggplot2)

# Read in Data
#-------------------------------------------
or <- read.csv("forplot.csv")
head(or)

# Cut Down and Clean Education Rows
#-------------------------------------------

#Subset
educ = or[1:5,]

#Rename Categories
educ$X2<- NA
educ$X2[educ$X == "educbin_Associates"] <- "Associates Degree"
educ$X2[educ$X == "educbin_Bachelors"] <- "Bachelors Degree"
educ$X2[educ$X == "educbin_GradDeg"] <- "Graduate Degree"
educ$X2[educ$X == "educbin_HS-grad"] <- "High School Degree"
educ$X2[educ$X == "educbin_Some-college"] <- "Some College"

# Boxplot!
#-------------------------------------------
positions <- c("High School Degree", "Some College", "Associates Degree", "Bachelors Degree", "Graduate Degree")

p <- ggplot(educ, aes(y=OR, x=X2)) +
     geom_point(colour = "purple4", size = 2, shape=15) + 
     geom_errorbar(aes(ymin=LB, ymax=UB, width=.2), color = "purple4", size = .7, linetype = "solid") +
     coord_flip() +
     geom_hline(yintercept = 1, colour = "grey40", linetype = "dashed") +
     labs(x = "Education Level", y = "Odds Ratio")+
     scale_y_continuous(breaks=c(seq(0,16,by=1))) +
     scale_x_discrete(limits = positions) 
p  


