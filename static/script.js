document.addEventListener(
    "DOMContentLoaded",
    function () {

        const form =
            document.querySelector("form");

        if (form) {

            form.addEventListener(
                "submit",
                function (event) {

                    const tenure =
                        document.querySelector(
                            'input[name="tenure"]'
                        ).value;

                    const monthly =
                        document.querySelector(
                            'input[name="monthly_charges"]'
                        ).value;

                    const total =
                        document.querySelector(
                            'input[name="total_charges"]'
                        ).value;


                    if (
                        tenure === "" ||
                        monthly === "" ||
                        total === ""
                    ) {

                        event.preventDefault();

                        alert(
                            "Please fill all required fields."
                        );

                        return;
                    }


                    if (
                        Number(tenure) < 0 ||
                        Number(monthly) < 0 ||
                        Number(total) < 0
                    ) {

                        event.preventDefault();

                        alert(
                            "Values cannot be negative."
                        );

                    }

                }
            );

        }

    }
);