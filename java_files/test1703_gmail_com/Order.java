package test1703_gmail_com; 

public class Order {
	
	public static String condition(String order) {
		double itemCost = 30.99;
		if(itemCost > 24.00) {
			order = "High Value Item!";
		} else {
			order = "Low Value Item!";
		}
		return order;
	}
	
	public static void main(String[] args) {
		String order = "";
		String result = condition(order);
		System.out.print(result);
	}
}