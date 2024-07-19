library(caret)
library(randomForest)
library(doParallel)
library(dplyr)

categoria_muestras = read.csv(file = "samples.csv", sep = "\t")

variantes_1pc = read.csv2(file = "dataset1_1pc.csv",sep = "\t")
variantes_1pc$grupo = categoria_muestras$class
variantes_1pc$grupo = as.factor(variantes_1pc$grupo)
rownames(variantes_1pc) = variantes_1pc[,1]
variantes_1pc = variantes_1pc[,-1]

variantes_1pc = variantes_1pc %>%
  select(last_col(), everything())

set.seed(1234)

zero_var = nzv(variantes_1pc)
variantes_1pc = variantes_1pc[,-zero_var]

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

resultado_rfe_rf = rfe(x = x_train_prueba,
                       y = y_train_prueba,
                       sizes = c(2:664),
                       rfeControl = control_rfe_rf)

tiempo.fin = Sys.time() - tiempo
print(tiempo.fin)

stopCluster(cl)

saveRDS(resultado_rfe_rf, file = "resultado_rfe_rf_664")



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
                            sizes = c(2:664),
                            rfeControl = control_rfe_treebag)

tiempo.fin = Sys.time() - tiempo
print(tiempo.fin)

stopCluster(cl)

saveRDS(resultado_rfe_treebag, file = "resultado_rfe_treebag_664")

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
                       sizes = c(2:664),
                       rfeControl = control_rfe_nb)

tiempo.fin = Sys.time() - tiempo
print(tiempo.fin)

stopCluster(cl)

saveRDS(resultado_rfe_nb, file = "resultado_rfe_nb_664")

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
                        sizes = c(2:664),
                        rfeControl = control_rfe_lda)

tiempo.fin = Sys.time() - tiempo
print(tiempo.fin)

stopCluster(cl)

saveRDS(resultado_rfe_lda, file = "resultado_rfe_lda_664")
