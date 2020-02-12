### Roberto Minini
# Computer Vision Challenge Solution
### Task 1
The function for solving the first task is **approximate_centers**. To solve the problem I first filtered the red pixel in the picture, using two different bound for the color. Then using the mask just created i stacked the points together and i ran the sklearn KMeans algorithm on these points. The function returns the approximated coordinates of the centers. Moreover, it plots the initial image together with blue points which represent the centers of each red dot.

The main challenged I faced here was filtering the pixels using the HSV scale and properly stacking the coordinates of the points together

### Bonus question
The function to solve the Bonus Question is **pair_points**. The function first finds the approximated coordinates of the centers of the two sets of red points. (call them OldCenters). Then the function runs the KMeans algorithm on all the red points from the two sets. In these way (say that each set contains four points) 4 new centers will be created (call them NewCenters) and the two OldCenters belonging to each corner will be associated to the corresponding NewCenter representing the corner. The function then checks the labels of each OldCenter and if two OldCenters share the same label they get paired. The function also plots lines to link the two points belonging to the same corner.

The main challenge here was creating a function which can properly know the position of each OldCenter inside the array containing all of the points. This was fundamental in order to get the label of the OldCenter which as was previously explained is used to pair it to the other OldCenter belonging to the same corner.
