library(caret)
library(randomForest)
library(doParallel)
library(dplyr)

categoria_muestras = read.csv(file = "samples.csv", sep = "\t")

variantes_1pc = read.csv2(file = "dataset2_5pc.csv",sep = "\t")
variantes_1pc$grupo = categoria_muestras$class
variantes_1pc$grupo = as.factor(variantes_1pc$grupo)
rownames(variantes_1pc) = variantes_1pc[,1]
variantes_1pc = variantes_1pc[,-1]

variantes_1pc = variantes_1pc %>%
  select(last_col(), everything())

set.seed(1234)

particion = createDataPartition(variantes_1pc$grupo, p = .80, list = FALSE)

train_prueba = variantes_1pc[particion, ]
test_prueba = variantes_1pc[-particion, ]

x_train_prueba = train_prueba[, -1]
y_train_prueba = train_prueba[, 1]

control_rfe_rf = rfeControl(functions = rfFuncs,
                            method = "repeatedcv",
                            repeats = 2,
                            number = 5,
                            allowParallel = TRUE)

set.seed(1234)

tiempo = Sys.time()

cl = makeCluster(20)
registerDoParallel(cl)

resultado_rfe_rf_ = rfe(x = x_train_prueba,
                       y = y_train_prueba,
                       sizes = c(1:20,40,60,80,100,150,200,250,300,350,400,500,600,700,800,900,1000,1250,1500,1750,2000,2250,2500,2750,3000,3250,3500,3750,4000),
                       rfeControl = control_rfe_rf)

tiempo.fin = Sys.time() - tiempo
print(tiempo.fin)

stopCluster(cl)

saveRDS(resultado_rfe_rf, file = "resultado_rfe_rf_6000_sinzv")



control_rfe_treebag = rfeControl(functions = treebagFuncs,
                                 method = "repeatedcv",
                                 repeats = 2,
                                 number = 5,
                                 allowParallel = TRUE)


set.seed(1234)

tiempo = Sys.time()

cl = makeCluster(20)
registerDoParallel(cl)

resultado_rfe_treebag = rfe(x = x_train_prueba,
                            y = y_train_prueba,
                            sizes = c(1:20,40,60,80,100,150,200,250,300,350,400,500,600,700,800,900,1000,1250,1500,1750,2000,2250,2500,2750,3000,3250,3500,3750,4000),
                            rfeControl = control_rfe_treebag)

tiempo.fin = Sys.time() - tiempo
print(tiempo.fin)

stopCluster(cl)

saveRDS(resultado_rfe_treebag, file = "resultado_rfe_treebag_6000_sinzv")

control_rfe_nb = rfeControl(functions = nbFuncs,
                            method = "repeatedcv",
                            repeats = 2,
                            number = 5,
                            allowParallel = TRUE)

set.seed(1234)

tiempo = Sys.time()

cl = makeCluster(20)
registerDoParallel(cl)

resultado_rfe_nb = rfe(x = x_train_prueba,
                       y = y_train_prueba,
                       sizes = c(1:20,40,60,80,100,150,200,250,300,350,400,500,600,700,800,900,1000,1250,1500,1750,2000,2250,2500,2750,3000,3250,3500,3750,4000),
                       rfeControl = control_rfe_nb)

tiempo.fin = Sys.time() - tiempo
print(tiempo.fin)

stopCluster(cl)

saveRDS(resultado_rfe_nb, file = "resultado_rfe_nb_6000_sinzv")

control_rfe_lda = rfeControl(functions = ldaFuncs,
                             method = "repeatedcv",
                             repeats = 2,
                             number = 5,
                             allowParallel = TRUE)

set.seed(1234)

tiempo = Sys.time()

cl = makeCluster(20)
registerDoParallel(cl)

resultado_rfe_lda = rfe(x = x_train_prueba,
                        y = y_train_prueba,
                        sizes = c(1:20,40,60,80,100,150,200,250,300,350,400,500,600,700,800,900,1000,1250,1500,1750,2000,2250,2500,2750,3000,3250,3500,3750,4000),
                        rfeControl = control_rfe_lda)

tiempo.fin = Sys.time() - tiempo
print(tiempo.fin)

stopCluster(cl)

saveRDS(resultado_rfe_lda, file = "resultado_rfe_lda_6000_sinzv")
