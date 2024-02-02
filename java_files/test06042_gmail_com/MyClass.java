package test06042_gmail_com; 

public class MyClass {
    
    public static double add(int x, double y) {
        
        double result = x + y ;
        return result;
    }
    
    public static void main(String args[]) {
        
        int x = 20;
        double y = 10.5;
        double result = add(x,y);
        System.out.print(result);
    }
}